// Copyright 2020 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// This seems to be bug in the compiler.
#[macro_use]
extern crate clap;
#[macro_use]
extern crate log;
extern crate llvm_sys as llvm;
//extern crate readmem;

use anyhow::{bail, Context, Result};
use clap::Arg;
use llvm_sys::{
    bit_writer::*, core::*, execution_engine::*, initialization::*, support::*, target::*,
};
//use readmem::{readmem, ContentType};
use std::{
    ffi::CString, 
    os::raw::c_int, 
    path::Path, 
    ptr::null_mut, 
    fs::File,
    str::FromStr,
    collections::HashMap,
    fs,
    io::{Write, Seek, Read, BufReader, ErrorKind, SeekFrom},
    io::prelude::*,
    num::ParseIntError
};

pub mod bootroms;
pub mod configuration;
pub mod engine;
pub mod peripherals;
pub mod riscv;
mod runtime;
mod softfloat;
pub mod tran;
pub mod util;
pub mod readmem;
pub mod dram_preload;

use crate::configuration::*;
use crate::engine::*;
use crate::readmem::{readmem, ContentType};
use crate::dram_preload::*;

use bytebuffer::ByteBuffer;
use to_binary::{BinaryString,BinaryError};
use byteorder::{BigEndian, ReadBytesExt, LittleEndian};

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
        // INFO: VIVI edit
        .arg(
            Arg::with_name("train-data-file-path")
                .long("train-data-file-path")
                .takes_value(true)
                .help("Path to data files (without labels) for training."),
        )
        .arg(
            Arg::with_name("train-labels-file-path")
                .long("train-labels-file-path")
                .takes_value(true)
                .help("Path to the labels of the data for training."),
        )
        .arg(
            Arg::with_name("train-data-bin-file-path")
                .long("train-data-bin-file-path")
                .takes_value(true)
                .help("Path to the binary file of the data for training."),
        )
        .arg(
            Arg::with_name("train-data-mem-offset")
                .long("train-data-mem-offset")
                .takes_value(true)
                .help("Define the offset at which the train data should be written into DRAM."),
        )
        .arg(
            Arg::with_name("train-labels-bin-file-path")
                .long("train-labels-bin-file-path")
                .takes_value(true)
                .help("Path to the binary file of the labels for training."),
        )
        .arg(
            Arg::with_name("train-labels-mem-offset")
                .long("train-labels-mem-offset")
                .takes_value(true)
                .help("Define the offset at which the train labels should be written into DRAM."),
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
    // INFO: VIVI edit 
    // let has_train_data = matches.is_present("train-data-file-path");
    // let has_train_labels = matches.is_present("train-labels-file-path");
    let has_train_bin = matches.is_present("train-data-bin-file-path");
    let has_train_labels_bin = matches.is_present("train-labels-bin-file-path");

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
    /* Preload the DRAM with the dataset 
    *  provided in the SnitchUtilities CMAKE
    *  file.
    */
    // First we check whether a file containing the training data (without labels)
    // has been provided.
    /*if has_train_data {
        trace!("Entering DRAM preloading");
        let train_data_path = matches.value_of("train-data-file-path").unwrap();
        // we save the contents of the file or throw an error if something went wrong
        // TODO: find more efficient alternative to read string
        let train_data_contents = fs::read_to_string(train_data_path)
            .expect("Something went wrong reading the file");
        // // now we read in the contents into a HashMap<u64, u32> as <address, value> pair
        let train_data = readmem::<u32>(&train_data_contents, ContentType::Float).unwrap();
        // get memory handle, mutex thread lock, unwrap checks if element found, otherwise panics
        // memory is defined as HashMap --> memory: Mutex<HashMap<u64, u32>>
        // where the first value corresponds to the address 
        let mut mem = engine.memory.lock().unwrap();
        // we will extend the DRAM by the values we just read in
        trace!("Starting to write the training data to DRAM");
        mem.extend(train_data);
        for i in 0x80040000..0x80104000 {
            let val = mem.get(&(i)).copied().unwrap_or(0);
            let float = unsafe { std::mem::transmute::<u32, f32>(val) };
            trace!("address = 0x{:x}, original value = {:b}, float value = {}", i, val, float);
        }
    }*/

    // First we check whether a file containing the training data (without labels)
    // has been provided.
    /*if has_train_labels {
        trace!("Loading labels");
        let train_labels_path = matches.value_of("train-labels-file-path").unwrap();
        // we save the contents of the file or throw an error if something went wrong
        let train_labels_contents = fs::read_to_string(train_labels_path)
            .expect("Something went wrong reading the file");
        // now we read in the contents into a HashMap<u64, u32> as <address, value> pair
        let train_labels = readmem::<u32>(&train_labels_contents, ContentType::Hex).unwrap();
        // get memory handle, mutex thread lock, unwrap checks if element found, otherwise panics
        // memory is defined as HashMap --> memory: Mutex<HashMap<u64, u32>>
        // where the first value corresponds to the address 
        let mut mem = engine.memory.lock().unwrap();
        // we will extend the DRAM by the values we just read in
        trace!("Starting to write the training labels to DRAM");
        mem.extend(train_labels);
        // for (k, v) in mem.iter(){
        //     trace!("address = 0x{:x}, HEX value = 0x{:x}, value = {}", k, v, v);
        // }
        for i in 0x80108000..0x80108400 {
            let val:u32 = mem.get(&(i)).copied().unwrap_or(0);
            // let float = unsafe { std::mem::transmute::<u32, f32>(val) };
            trace!("address = 0x{:x}, HEX value = 0x{:x}, label value = {}", i, val, val);
        }
    }*/

    if has_train_bin {

        
        let bin_path = matches.value_of("train-data-bin-file-path").unwrap();
        
        trace!("Loading train data from binary file: {}", bin_path);

        // get memory offset from argument
        let mut memory_offset = matches.value_of("train-data-mem-offset").unwrap().trim_start_matches("0x");
        // turn the string into a u64
        let mut mem_offset = u64::from_str_radix(memory_offset, 16).unwrap();

        trace!("Train data starts at address: 0x{:x}", mem_offset);

        let train_data = dram_preload::bin_read(bin_path, mem_offset).unwrap();
            
        let train_data_length = train_data.len() as u64;

        let mut mem = engine.memory.lock().unwrap();
        
        mem.extend(train_data);

        for addr in mem_offset .. mem_offset + train_data_length {
            let val:u32 = mem.get(&(addr)).copied().unwrap_or(0);
            trace!("address = 0x{:x}, binary value = {:#034b}", addr, val);
        }

    }

    if has_train_labels_bin {
        let bin_path = matches.value_of("train-labels-bin-file-path").unwrap();
        trace!("Loading train labels from binary file: {}", bin_path);
        // get memory offset from argument
        let mut memory_offset = matches.value_of("train-labels-mem-offset").unwrap().trim_start_matches("0x");
        // turn the string into a u64
        let mut mem_offset = u64::from_str_radix(memory_offset, 16).unwrap();
        trace!("Train labels starts at address: 0x{:x}", mem_offset);
        let dtype = "U32";
        let train_labels = dram_preload::bin_u32_read(bin_path, mem_offset).unwrap();
        let train_labels_length = train_labels.len() as u64;
        let mut mem = engine.memory.lock().unwrap();
        mem.extend(train_labels);
        for addr in mem_offset .. mem_offset + train_labels_length {
            let val:u32 = mem.get(&(addr)).copied().unwrap_or(0);
            trace!("address = 0x{:x}, binary value = {:#034b}", addr, val);
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
