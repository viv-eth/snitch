while getopts :c:r:t:d:f:h opt; do
  case "${opt}" in
    h)
      echo "[RTL build] Usage: sim_build.sh [-h]"
      echo "[RTL build]   -h: Display this help message"
      echo "[RTL build] Usage: sim_build.sh [-c]"
      echo "[RTL build]   -c: Select which cmake version to use for run. Default is 3.18.1"
      echo "[RTL build] Usage: sim_build.sh [-r]"
      echo "[RTL build]   -r: Select the snRuntime (banshee or cluster). Default is cluster"
      echo "[RTL build] Usage: sim_build.sh [-t]"
      echo "[RTL build]   -t: Select the toolchain. Default is llvm"
      echo "[RTL build] Usage: sim_build.sh [-d]"
      echo "[RTL build]   -d: Set to one to remove all existing files in the build folder."
      echo "[RTL build] Usage: sim_build.sh [-f]"
      echo "[RTL build]   -f: Select if only a specific binary should be built. If not set all binaries will be built."
      exit 1
      ;;
    c)
      version="${OPTARG}" # 3.18.1
      echo "[RTL build] Using CMAKE version $version"
      ;;
    r)
        runtime="${OPTARG}" # snRuntime-cluster
        echo "[RTL build] Using runtime $runtime"
        ;;
    t)
        toolchain="${OPTARG}" # toolchain-llvm
        echo "[RTL build] Using toolchain $toolchain"
        ;;
    d)
        remove_all="${OPTARG}"
        if [ "$remove_all" = "1" ]; then
            echo "[RTL build] Removing following files from the build folder:"
            sh_pattern="*.sh"
            for i in *; do
                if ([[ $i != $sh_pattern ]]); then
                    echo "[RTL build] Removing $i"
                    rm -rf $i
                fi
            done
        else
            echo "[RTL build] Files have not been removed."
        fi
        ;;
    f)
        binary="${OPTARG}"
        echo "[RTL build] Building binary $binary"
        ;;
    \?)
      echo "[RTL build] Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

START_D=$( date "+%d/%m/%y" )
START_H=$( date "+%H:%M:%S" )
echo "[HW] Setting up binaries for new simulation at: $START_H on $START_D"

# assign default values to variables if not set
if [ -z $version ]; then
    version="3.18.1"
fi

if [ -z $runtime ]; then
    runtime="cluster"
fi

if [ -z $toolchain ]; then
    toolchain="llvm"
fi

cmake-$version -DSNITCH_RUNTIME=snRuntime-$runtime -DCMAKE_TOOLCHAIN_FILE=toolchain-$toolchain ../

if [ -z $binary ]
  then
    echo "[RTL build] Building all binaries"
    make -j
else
echo "[RTL build] Building binary $binary"
make -w $binary
fi

END_D=$( date "+%d/%m/%y" )
END_H=$( date "+%H:%M:%S" )
echo "[HW] Finished build at: $END_H on $END_D"

