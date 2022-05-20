// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// This seems to be bug in the compiler.
#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
extern crate llvm_sys as llvm;

use anyhow::{bail, Context, Result};
use clap::Arg;
use llvm_sys::{
    bit_writer::*, core::*, execution_engine::*, initialization::*, support::*, target::*,
};
use std::{ffi::CString, os::raw::c_int, path::Path, ptr::null_mut, fs::File};

pub mod bootroms;
pub mod configuration;
pub mod engine;
pub mod peripherals;
pub mod riscv;
mod runtime;
mod softfloat;
pub mod tran;
pub mod util;

use crate::configuration::*;
use crate::engine::*;

fn main() -> Result<()> {
    // Parse the command line arguments.
    let matches = app_from_crate!()
        .arg(
            Arg::with_name("binary")
                .help("RISC-V ELF binary to execute")
                .required(true),
        )
        .arg(
            Arg::with_name("dump-llvm")
                .long("dump-llvm")
                .short("d")
                .help("Dump the translated LLVM IR module"),
        )
        .arg(
            Arg::with_name("emit-llvm")
                .long("emit-llvm")
                .short("S")
                .takes_value(true)
                .help("Emit the translated LLVM assembly to a file"),
        )
        .arg(
            Arg::with_name("emit-bitcode")
                .long("emit-bitcode")
                .short("c")
                .takes_value(true)
                .help("Emit the translated LLVM bitcode to a file"),
        )
        .arg(
            Arg::with_name("dry-run")
                .long("dry-run")
                .short("n")
                .help("Translate the binary, but do not execute"),
        )
        .arg(
            Arg::with_name("no-opt-llvm")
                .long("no-opt-llvm")
                .help("Do not optimize LLVM IR (default, deprecated)"),
        )
        .arg(
            Arg::with_name("no-opt-jit")
                .long("no-opt-jit")
                .help("Do not optimize during JIT compilation (default, deprecated)"),
        )
        .arg(
            Arg::with_name("opt-llvm")
                .long("opt-llvm")
                .help("Optimize LLVM IR"),
        )
        .arg(
            Arg::with_name("opt-jit")
                .long("opt-jit")
                .help("Optimize during JIT compilation"),
        )
        .arg(
            Arg::with_name("trace")
                .long("trace")
                .short("t")
                .help("Enable instruction tracing"),
        )
        .arg(
            Arg::with_name("no-interrupt")
                .long("no-interrupt")
                .help("Disable interrupt support for faster execution"),
        )
        .arg(
            Arg::with_name("latency")
                .long("latency")
                .short("l")
                .help("Enable instruction latency modeling"),
        )
        .arg(
            Arg::with_name("num-cores")
                .long("num-cores")
                .takes_value(true)
                .help("Number of cores to simulate"),
        )
        .arg(
            Arg::with_name("num-clusters")
                .long("num-clusters")
                .takes_value(true)
                .help("Number of clusters to simulate"),
        )
        .arg(
            Arg::with_name("configuration")
                .long("configuration")
                .takes_value(true)
                .help("A configuration file describing the architecture"),
        )
        .arg(
            Arg::with_name("create-configuration")
                .long("create-configuration")
                .takes_value(true)
                .help("Write the default configuration to this file"),
        )
        .arg(
            Arg::with_name("base-hartid")
                .long("base-hartid")
                .takes_value(true)
                .help("The hartid of the first core"),
        )
        .arg(
            Arg::with_name("llvm-args")
                .short("L")
                .takes_value(true)
                .multiple(true)
                .help("Pass command line arguments to LLVM"),
        )
        // INFO: VIVI edit --> approved
        .arg(
            Arg::with_name("train-file-path")
                .long("train-file-path")
                .takes_value(true)
                .help("Path to data files for training."),
        )
        .arg(
            Arg::with_name("start-address")
                .long("start-address")
                .takes_value(true)
                .help("Start address where to store dataset DRAM."),
        )
        .get_matches();

    // Configure the logger.
    pretty_env_logger::init_custom_env("SNITCH_LOG");

    // Initialize the LLVM core.
    let context = unsafe {
        LLVMLinkInMCJIT();
        LLVM_InitializeNativeTarget();
        LLVM_InitializeNativeAsmPrinter();

        // Initialize passes (inspired by llvm/tools/opt/opt.cpp:527).
        let pass_reg = LLVMGetGlobalPassRegistry();
        LLVMInitializeAggressiveInstCombiner(pass_reg);
        LLVMInitializeAnalysis(pass_reg);
        LLVMInitializeCodeGen(pass_reg);
        LLVMInitializeCore(pass_reg);
        LLVMInitializeIPA(pass_reg);
        LLVMInitializeIPO(pass_reg);
        LLVMInitializeInstCombine(pass_reg);
        LLVMInitializeInstrumentation(pass_reg);
        LLVMInitializeObjCARCOpts(pass_reg);
        LLVMInitializeScalarOpts(pass_reg);
        LLVMInitializeTarget(pass_reg);
        LLVMInitializeTransformUtils(pass_reg);
        LLVMInitializeVectorization(pass_reg);

        engine::add_llvm_symbols();
        LLVMGetGlobalContext()
    };

    // Pass command line arguments to LLVM.
    if let Some(args) = matches.values_of("llvm-args") {
        let exec_name = CString::new("banshee").unwrap();
        let args: Vec<_> = args.map(|a| CString::new(a).unwrap()).collect();
        let mut argv = vec![];
        argv.push(exec_name.as_ptr());
        argv.extend(args.iter().map(|a| (*a).as_ptr()));
        let overview = CString::new("Banshee is magic!").unwrap();
        unsafe {
            LLVMParseCommandLineOptions(
                argv.len() as c_int,
                argv.as_ptr(),
                overview.as_ptr() as *const _,
            );
        }
    }

    // Setup the execution engine.
    let mut engine = Engine::new(context);
    if matches.is_present("opt-llvm") && matches.is_present("no-opt-llvm") {
        bail!("Both --opt-llvm and --no-opt-llvm provided");
    }
    if matches.is_present("opt-jit") && matches.is_present("no-opt-jit") {
        bail!("Both --opt-jit and --no-opt-jit provided");
    }
    engine.opt_llvm = matches.is_present("opt-llvm");
    engine.opt_jit = matches.is_present("opt-jit");
    engine.interrupt = !matches.is_present("no-interrupt");
    if engine.interrupt {
        debug!("Interrupts enabled");
    }
    engine.trace = matches.is_present("trace");
    engine.latency = matches.is_present("latency");

    let has_num_cores = matches.is_present("num-cores");
    let has_num_clusters = matches.is_present("num-clusters");
    let has_base_hartid = matches.is_present("base-hartid");
    // INFO: VIVI edit --> approved
    let has_train_file_path = matches.is_present("train-file-path");
    let has_start_addr = matches.is_present("start-address");

    matches
        .value_of("num-cores")
        .map(|x| engine.num_cores = x.parse().unwrap());
    matches
        .value_of("num-clusters")
        .map(|x| engine.num_clusters = x.parse().unwrap());
    matches
        .value_of("base-hartid")
        .map(|x| engine.base_hartid = x.parse().unwrap());

    if let Some(file) = matches.value_of("create-configuration") {
        Configuration::print_default(file)?;
    }
    debug!("Configuration used:\n{}", engine.config);
    // debug!("Configuration used: {} {} {}\n", engine.num_cores, engine.num_clusters, engine.base_hartid);

    engine.config = if let Some(config_file) = matches.value_of("configuration") {
        // if configuration file is given and `architecture` information is set
        // use that configuration, else banshee parameter
        let config_used: Configuration = Configuration::parse(
            config_file,
            engine.num_clusters,
            has_num_clusters,
            engine.num_cores,
            has_num_cores,
            engine.base_hartid,
            has_base_hartid,
        );
        // get configuration
        engine.num_cores = config_used.architecture.num_cores;
        engine.num_clusters = config_used.architecture.num_clusters;
        engine.base_hartid = config_used.architecture.base_hartid;
        config_used
    } else {
        Configuration::new(engine.num_clusters, engine.num_cores, engine.base_hartid)
    };
    debug!("Configuration used:\n{}", engine.config);

    // Read the binary.
    let path = Path::new(matches.value_of("binary").unwrap());
    info!("Loading binary {}", path.display());
    let elf = match elf::File::open_path(&path) {
        Ok(f) => f,
        Err(e) => bail!("Failed to open binary {}: {:?}", path.display(), e),
    };

    // Create a module for each cluster
    engine.create_modules();

    // Translate the binary.
    engine
        .translate_elf(&elf)
        .context("Failed to translate ELF binary")?;

    // INFO: VIVI edit --> approved
    // preload the DRAM with the dataset 
    // path to dataset and start address 
    // First we check whether there exists a path to the dataset
    if has_train_file_path {
        // get the path from the config file
        let dataset_path = matches.value_of("train-file-path").unwrap();
        // open & parse the file
        let dataset_file = File::open(dataset_path)?;
        let mut rdr = csv::Reader::from_reader(dataset_file); // INFO: maybe change to read from path
        // get memory handle, mutex thread lock, unwrap checks if element found, otherwise panics
        // memory is defined as HashMap --> memory: Mutex<HashMap<u64, u32>>
        // where the first value corresponds to the address 
        let mut mem = engine.memory.lock().unwrap();
        // get the start address of the dataset file (NOTE: make sure to not overwrite your ELF data!! very bad)
        // the start address has to be defined in the config file
        let mut addr: u64 = u64::from_str_radix(matches.value_of("start-address").unwrap().trim_start_matches("0x"), 16).unwrap();
        // iterate through the records (i.e. rows) of the dataset CSV file
        // CSV file format: label followed by data in a row
        use byteorder::{LittleEndian, ReadBytesExt};
        for result in rdr.records() {
            let record = result?;
            // get the number of data items in a record
            let record_length = record.len();
            // first item in a row is the label --> panic if we don't have a label
            let label: u32 = record[0].parse()?; // record reads in strings, but will automatically parse to defined type <u32>
            // now we write the label together with its corresponding address into memory
            mem.insert(
                addr, label
            );
            trace!("  - 0x{:x} = 0x{:x}", addr, label);
            // after each insert we have to increment the address to the next writable position
            addr += 0x4;
            for pixel_index in 1..record_length{
                // let pixel: f32 = record[pixel_index].parse()?; // truncate the value too f32 instead of f64 for le conversion
                // // save pixel value as little endian unsigned integer representation
                // let pixel_bytes = pixel.to_le_bytes(); // this returns [u8, 4] array for f32 & [u8, 8] array for f64
                // // Apparently what I am about to do is *extremely* unsafe:
                // // We want to represent the floating point 32-bit number as a u32
                // let pixel_u32 = unsafe {
                //     std::mem::transmute::<[u8; 4], u32>(pixel_bytes) // turn raw bytes into u32 number
                // };
                // to avoid the incredibly rebellious thing I did above I read in the image data as unsigned ints
                // directly and normalize in the network script after reading from memory --> TODO: check latency
                let pixel_u32: u32 = record[pixel_index].parse()?;
                mem.insert( 
                    addr, pixel_u32
                );
                trace!("  - 0x{:x} = 0x{:x}", addr, pixel_u32);
                addr += 0x4;
            }
        }
        
        
    }

    // Write the module to disk if requested.
    if let Some(path) = matches.value_of("emit-llvm") {
        unsafe {
            LLVMPrintModuleToFile(
                engine.modules[0],
                format!("{}\0", path).as_ptr() as *const _,
                null_mut(),
            );
        }
    }
    if let Some(path) = matches.value_of("emit-bitcode") {
        unsafe {
            LLVMWriteBitcodeToFile(
                engine.modules[0],
                format!("{}\0", path).as_ptr() as *const _,
            );
        }
    }

    // Dump the module if requested.
    if matches.is_present("dump-llvm") {
        unsafe {
            LLVMDumpModule(engine.modules[0]);
        }
    }

    // Init the peripherals
    engine.init_periphs();

    // Init the Bootrom
    engine.init_bootrom();

    // Execute the binary.
    if !matches.is_present("dry-run") {
        let return_code = engine.execute().context("Failed to execute ELF binary")?;
        std::process::exit(return_code as i32);
    }
    Ok(())
}
