echo "Setup Isaac Sim Conda environment."
echo "Isaac Sim path: ${ISAACSIM_PATH}"

export PYTHONPATH_PREV=$PYTHONPATH
export LD_LIBRARY_PATH_PREV=$LD_LIBRARY_PATH

source ${ISAACSIM_PATH}/setup_conda_env.sh
