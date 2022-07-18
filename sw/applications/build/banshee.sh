while getopts :c:r:t:d:f:h opt; do
  case "${opt}" in
    h)
      echo "[BANSHEE] Usage: sim_build.sh [-h]"
      echo "[BANSHEE]   -h: Display this help message"
      echo "[BANSHEE] Usage: sim_build.sh [-c]"
      echo "[BANSHEE]   -c: Select which cmake version to use for run. Default is 3.18.1"
      echo "[BANSHEE] Usage: sim_build.sh [-r]"
      echo "[BANSHEE]   -r: Select the snRuntime (banshee or cluster). Default is banshee"
      echo "[BANSHEE] Usage: sim_build.sh [-t]"
      echo "[BANSHEE]   -t: Select the toolchain. Default is llvm"
      echo "[BANSHEE] Usage: sim_build.sh [-d]"
      echo "[BANSHEE]   -d: Set to one to remove all existing files in the build folder."
      echo "[BANSHEE] Usage: sim_build.sh [-f]"
      echo "[BANSHEE]   -f: Select if only a specific binary should be built. If not set all binaries will be built."
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
    \?)
      echo "[BANSHEE] Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

START_D=$( date "+%d/%m/%y" )
START_H=$( date "+%H:%M:%S" )
echo "[BANSHEE] Building binaries for new simulation at: $START_H on $START_D"

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

start_time=$SECONDS
cmake-$version -DSNITCH_RUNTIME=snRuntime-$runtime -DCMAKE_TOOLCHAIN_FILE=toolchain-$toolchain ../

if [ -z $binary ]
  then
    echo "[BANSHEE] Building all binaries"
    make -j
else
echo "[BANSHEE] Building binary $binary"
make -w $binary
fi

elapsed=$(( SECONDS - start_time ))

END_D=$( date "+%d/%m/%y" )
END_H=$( date "+%H:%M:%S" )

echo "[BANSHEE] Finished build at: $END_H on $END_D"


eval "echo [BANSHEE] Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

