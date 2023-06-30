JOB_COMMAND=${1:-echo hi}

# MAX_MEMORY is the maximum number of parallel jobs fitting in the GPU
MAX_MEMORY=${2:-30000}

# SLEEP_TIME is the time to sleep between memory checks
SLEEP_TIME=${3:-60}

# DEBUG is a flag that we hardcode
DEBUG=0
if [ $DEBUG -ge 1 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M')] Executing: ${JOB_COMMAND}"
  exit
fi

# Check memory and wait until ready
CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
echo "[$(date '+%Y-%m-%d %H:%M')] Found ${CURR_MEMORY} MB out of ${TOTAL_MEMORY} MB."
while (( $CURR_MEMORY > $MAX_MEMORY )); do
  echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping as ${CURR_MEMORY} MB is greater than ${MAX_MEMORY} MB."
  sleep ${SLEEP_TIME}
  CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
  TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
done
echo "[$(date '+%Y-%m-%d %H:%M')] Executing: ${JOB_COMMAND}"
${JOB_COMMAND} &

# Wait until the job appears in nvidia-smi
PROCESS_PID=$!
COUNTER=0
while [ `nvidia-smi | grep $PROCESS_PID | wc -l` -ne "1" ]; do
  echo "[$(date '+%Y-%m-%d %H:%M')] Waiting for ${PROCESS_PID} PID to show in nvidia-smi. Sleeping (Try: ${COUNTER})."
  sleep ${SLEEP_TIME}
  COUNTER=$((COUNTER + 1))
  if [ $COUNTER -ge 30 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M')] Exiting as the job has not appeared in nvidia-smi after ${COUNTER} tries."
    break
  fi
done
