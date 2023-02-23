JOB_COMMAND=${1:-echo hi}

# MAX_MEMORY is the maximum number of parallel jobs fitting in the GPU
MAX_MEMORY=${2:-30000}

# SLEEP_TIME is the time to sleep between memory checks
SLEEP_TIME=${3:-60}

# Check memory and wait until ready
CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
echo "[$(date '+%Y-%m-%d %H:%M')] Found ${CURR_MEMORY} MB out of ${TOTAL_MEMORY} MB."
while (( $CURR_MEMORY > $MAX_MEMORY )); do
  echo "[$(date '+%Y-%m-%d %H:%M')] Sleeping as ${CURR_MEMORY} MB is greater than ${MAX_MEMORY} MB."
  sleep 15
  CURR_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f1 | sed 's/MiB//g' | sed 's/ //g'`
  TOTAL_MEMORY=`nvidia-smi | grep -E '([0-9]+MiB) */ *([0-9]+MiB)' | sed 's/.* \([0-9]\+MiB *\/ *\+[0-9]\+MiB\).*/\1/g' | cut -d'/' -f2 | sed 's/MiB//g' | sed 's/ //g'`
done
echo "[$(date '+%Y-%m-%d %H:%M')] Executing: ${JOB_COMMAND}"
${JOB_COMMAND} &
