while getopts :c:r:t:d:f:v:s:l:h opt; do
  case "${opt}" in
    h)
      echo "[BANSHEE] Usage: banshee.sh [-h]"
      echo "[BANSHEE]   -h: Display this help message"
      echo "[BANSHEE] Usage: banshee.sh [-c]"
      echo "[BANSHEE]   -c: Select which cmake version to use for run. Default is 3.18.1"
      echo "[BANSHEE] Usage: banshee.sh [-r]"
      echo "[BANSHEE]   -r: Select the snRuntime (banshee or cluster). Default is banshee"
      echo "[BANSHEE] Usage: banshee.sh [-t]"
      echo "[BANSHEE]   -t: Select the toolchain. Default is llvm"
      echo "[BANSHEE] Usage: banshee.sh [-d]"
      echo "[BANSHEE]   -d: Set to one to remove all existing files in the build folder."
      echo "[BANSHEE] Usage: banshee.sh [-f]"
      echo "[BANSHEE]   -f: Select if only a specific binary should be built. If not set all binaries will be built."
      echo "[BANSHEE] Usage: banshee.sh [-v]"
      echo "[BANSHEE]   -v: Define whether verbose output should be enabled. Default is disabled."
      echo "[BANSHEE] Usage: banshee.sh [-s]"
      echo "[BANSHEE]   -s: Define the SNITCH_LOG variable."
      echo "[BANSHEE] Usage: banshee.sh [-l]"
      echo "[BANSHEE]   -l: Define whether the output should be logged to a file or not."
      exit 1
      ;;
    c)
      version="${OPTARG}" # 3.18.1
      echo "[BANSHEE] Using CMAKE version $version"
      ;;
    r)
        runtime="${OPTARG}" # snRuntime-cluster
        echo "[BANSHEE] Using runtime $runtime"
        ;;
    t)
        toolchain="${OPTARG}" # toolchain-llvm
        echo "[BANSHEE] Using toolchain $toolchain"
        ;;
    d)
        remove_all="${OPTARG}"
        if [ "$remove_all" = "1" ]; then
            echo "[BANSHEE] Removing following files from the build folder:"
            sh_pattern="*.sh"
            for i in *; do
                if ([[ $i != $sh_pattern ]]); then
                    echo "[BANSHEE] Removing $i"
                    rm -rf $i
                fi
            done
        else
            echo "[BANSHEE] Files have not been removed."
        fi
        ;;
    f)
        binary="${OPTARG}"
        echo "[BANSHEE] Building binary $binary"
        ;;
    v) # verbose
        verbose="${OPTARG}"
        echo "[BANSHEE] Verbose mode is ${verbose}."
        ;;
    s) # SNITCH_LOG mode
        snitch_log_mode="${OPTARG}"
        echo "[BANSHEE] Setting SNITCH_LOG to $snitch_log_mode"
        ;;
    l) # define whether terminal output should be logged to file
        log="${OPTARG}"
        if [ "$log" = "1" ]; then
            echo "[BANSHEE] Logging terminal output to file."
        else
            echo "[BANSHEE] Terminal output will not be logged to file."
        fi
        ;;
    \?)
      echo "[BANSHEE] Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

START_D=$( date "+%d/%m/%y" )
START_H=$( date "+%H:%M:%S" )
echo "[BANSHEE] Building binaries for new simulation at: $START_H on $START_D"
export SNITCH_LOG=${snitch_log_mode}
echo "[BANSHEE] Setting SNITCH_LOG to $SNITCH_LOG"

# assign default values to variables if not set
if [ -z $version ]; then
    version="3.18.1"
fi

if [ -z $runtime ]; then
    runtime="banshee"
fi

if [ -z $toolchain ]; then
    toolchain="llvm"
fi

if [ -z $verbose ]; then
    verbose="OFF"
fi

# generate a random number for the build folder name
RANDOM_NUMBER=$(($RANDOM % 1000))


start_time=$SECONDS
cmake-$version -DSNITCH_RUNTIME=snRuntime-$runtime -DCMAKE_TOOLCHAIN_FILE=toolchain-$toolchain ../ 
if [ -z $binary ]
  then
    echo "[BANSHEE] Building all binaries"
    make -j
else
    echo "[BANSHEE] Building binary $binary"
    if [ "$log" = "1" ]; then
        filename=${binary}_banshee_${RANDOM_NUMBER}.txt
        # check if the file exists
        if [ -f $filename ]; then
            # change the name of the file by adding a random number
            RANDOM_NUMBER_2=$(($RANDOM % 1000))
            filename = ${binary}_banshee_${RANDOM_NUMBER + RANDOM_NUMBER_2}.txt
        fi
        make run-banshee-$binary VERBOSE=${verbose} 2>&1 | tee ${filename}
        # fi
        echo "[BANSHEE] Saving output to file: ${binary}_banshee_${RANDOM_NUMBER}.txt"
    else
        make run-banshee-$binary VERBOSE=${verbose}
    fi
fi

elapsed=$(( SECONDS - start_time ))

END_D=$( date "+%d/%m/%y" )
END_H=$( date "+%H:%M:%S" )

echo "[BANSHEE] Finished build at: $END_H on $END_D"


eval "echo [BANSHEE] Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

