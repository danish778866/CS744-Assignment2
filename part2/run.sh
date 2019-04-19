#!/bin/bash

SCRIPT=`basename ${BASH_SOURCE[0]}`
MODE=""
BATCH_SIZE=0 
SCRIPT_DIR=$(cd `dirname $0` && pwd)
UTILS_SCRIPT="${SCRIPT_DIR}/cluster_utils.sh"
LOGS_DIR="${SCRIPT_DIR}/logs"
if [ ! -d "${LOGS_DIR}" ]
then
    echo "Creating ${LOGS_DIR}..."
    mkdir ${LOGS_DIR}
fi
START_SERVER_SCRIPT="startserver.py"
START_SERVER_FUNC="start_cluster"
ALEXNET_TRAIN_SCRIPT="AlexNet.scripts.train"

# Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

# Help function
function HELP {
    exit_status=$1
    echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
    echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -m <execution_mode> -b <batch_size>${NORM}"\\n
    echo "Command line switches are required. The following switches are recognized."
    echo "${REV}-m${NORM}  --Sets the execution mode ${BOLD}m${NORM}. The possible values are single, cluster and cluster2"
    echo "${REV}-b${NORM}  --Sets the batch size ${BOLD}b${NORM}. Recommended values are one of 64, 128 or 256."
    echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
    echo -e "Example: ${BOLD}$SCRIPT -m cluster -b 64${NORM}"\\n
    exit $exit_status
}

# Check the number of arguments. If none are passed, print help and exit.
NUMARGS=$#
echo -e \\n"Number of arguments: $NUMARGS"
if [ $NUMARGS -eq 0 ]; then
    echo "Please specify the required arguments..."
    HELP 1
fi

### Start getopts code ###

#Parse command line flags
#If an option should be followed by an argument, it should be followed by a ":".
#Notice there is no ":" after "h". The leading ":" suppresses error messages from
#getopts. This is required to get my unrecognized option code to work.

while getopts :m:b:h FLAG; do
    case $FLAG in
        m)
            MODE=$OPTARG
            echo "-m used: $OPTARG"
            ;;
        b)
            BATCH_SIZE=$OPTARG
            echo "-b used: $OPTARG"
            ;;
        h)  #show help
            HELP 0
            ;;
        \?) #unrecognized option - show help
            echo -e \\n"Option -${BOLD}$OPTARG${NORM} not allowed."
            HELP 1
            ;;
    esac
done

shift $((OPTIND-1))  #This tells getopts to move on to the next argument.

### End getopts code ###

function validate_arguments {
    if [ -z "${MODE}" ]
    then
        echo "Please specify a valid execution mode..."
        HELP 1
    elif [ "${MODE}" != "single" ] && [ "${MODE}" != "cluster" ] && [ "${MODE}" != "cluster2" ]
    then
        echo "Please specify a valid execution mode..."
        HELP 1
    fi
    
    if [ "${BATCH_SIZE}" -eq 0 ]
    then
        echo "Please specify a valid batch size..."
        HELP 1
    fi
}

function execute_training {
    echo "Including functions from ${UTILS_SCRIPT}..."
    source ${UTILS_SCRIPT}
    echo "Starting cluster in ${MODE} mode..."
    "${START_SERVER_FUNC}" "${START_SERVER_SCRIPT}" "${MODE}"
    echo "Waiting for bootstrap of server processes..."
    sleep 5
    echo "Executing AlexNet training in ${MODE} with batch size of ${BATCH_SIZE}..."
    LOG_FILE="${LOGS_DIR}/AlexNet_${MODE}_${BATCH_SIZE}.log"
    python -m ${ALEXNET_TRAIN_SCRIPT} --mode ${MODE} --batch_size ${BATCH_SIZE} > ${LOG_FILE} 2>&1 &
    pid=$!
    echo "Training is running in process $pid. The progress can be viewed in log file ${LOG_FILE}..."
    echo "Waiting for the training process to complete..."
    wait $pid
    echo "Training is complete, terminating the cluster now..."
    terminate_cluster
    echo "Cluster has been terminated..."
}

validate_arguments
execute_training
exit 0
