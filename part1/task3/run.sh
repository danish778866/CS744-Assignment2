#!/bin/bash
export TF_RUN_DIR="~/tf"

SCRIPT=`basename ${BASH_SOURCE[0]}`
MODE=""
BATCH_SIZE=0 
SCRIPT_DIR=$(cd `dirname $0` && pwd)
LOGS_DIR="${SCRIPT_DIR}/logs"
if [ ! -d "${LOGS_DIR}" ]
then
    echo "Creating ${LOGS_DIR}..."
    mkdir ${LOGS_DIR}
fi
TRAIN_SCRIPT=""

# Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

# Help function
function HELP {
    exit_status=$1
    echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
    echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -m <execution_mode> -b <batch_size> -t <training_script>${NORM}"\\n
    echo "Command line switches are required. The following switches are recognized."
    echo "${REV}-m${NORM}  --Sets the execution mode ${BOLD}m${NORM}. The possible values are single, cluster and cluster2"
    echo "${REV}-b${NORM}  --Sets the batch size ${BOLD}b${NORM}. Recommended values are one of 32, 64, 128 or 256."
    echo "${REV}-t${NORM}  --Sets the python script ${BOLD}t${NORM} to be used for training."
    echo -e "${REV}-h${NORM}  --Displays this help message. No further functions are performed."\\n
    echo -e "Example: ${BOLD}$SCRIPT -m cluster -b 64 -t code_template.py${NORM}"\\n
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

while getopts :m:b:t:h FLAG; do
    case $FLAG in
        m)
            MODE=$OPTARG
            echo "-m used: $OPTARG"
            ;;
        b)
            BATCH_SIZE=$OPTARG
            echo "-b used: $OPTARG"
            ;;
        t)
            TRAIN_SCRIPT=$OPTARG
            echo "-t used: $OPTARG"
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

    if [ -z "${TRAIN_SCRIPT}" ]
    then
        echo "Please specify a valid python training script present in the current directory..."
        HELP 1
    elif [ ! -e "${SCRIPT_DIR}/${TRAIN_SCRIPT}" ]
    then
        echo "Please specify a valid python training script present in the current directory..."
        HELP 1
    fi
}


function terminate_cluster {
    echo "Terminating the servers..."
    CMD="ps aux | grep -v 'grep' | grep -v 'bash' | grep -v 'ssh' | grep 'python ${TRAIN_SCRIPT}' | awk -F' ' '{print \$2}' | xargs kill -9"
    echo "Executing $CMD..."
    for i in `seq 0 2`; do
        ssh node$i "$CMD"
    done
}


function install_tensorflow {
    for i in `seq 0 2`; do
        nohup ssh node$i "sudo apt update; sudo apt install --assume-yes python-pip python-dev; sudo pip install tensorflow"
    done
}


function start_cluster {
    echo "Creating $TF_RUN_DIR on remote hosts if they do not exist..."
    echo "Copying the script to all the remote hosts..."
    for i in `seq 0 2`; do
        ssh node$i "mkdir -p $TF_RUN_DIR"
        scp ${TRAIN_SCRIPT} node$i:$TF_RUN_DIR
    done
    echo "Starting tensorflow servers on all hosts based on the cluster specification in ${TRAIN_SCRIPT}..."
    echo "The server output is logged to serverlog-i.out, where i = 0, ..., 3 are the VM numbers..."
    if [ "${MODE}" = "single" ]
    then
        nohup ssh node0 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=single --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-0_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[0]=$!
    elif [ "${MODE}" = "cluster" ]
    then
        nohup ssh node0 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster  --job_name=ps" > "${LOGS_DIR}/serverlog-ps-0.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        nohup ssh node0 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster  --task_index=0 --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-0.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[0]=$!
        nohup ssh node1 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster  --task_index=1 --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-1.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[1]=$!
    else
        nohup ssh node0 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster2  --job_name=ps" > "${LOGS_DIR}/serverlog-ps-0.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        nohup ssh node0 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster2  --task_index=0 --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-0.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[0]=$!
        nohup ssh node1 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster2  --task_index=1 --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-1.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[1]=$!
        nohup ssh node2 "cd ~/tf ; python ${TRAIN_SCRIPT} --deploy_mode=cluster2  --task_index=2 --batch_size=${BATCH_SIZE}" > "${LOGS_DIR}/serverlog-2.out_${MODE}_${BATCH_SIZE}.log" 2>&1&
        pids[2]=$!
    fi
    echo "Started training processes with pid's ${pids[*]}..."
}

function wait_for_pids {
    echo "Waiting for all pid's to exit..."
    for pid in ${pids[*]}
    do
        wait $pid
    done
    echo "All pid's have exited..."
}

echo "Validating arguments..."
validate_arguments
echo "Arguments validated..."
echo "Starting the cluster of training jobs..."
start_cluster
wait_for_pids
terminate_cluster
