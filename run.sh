#!/bin/bash -l

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# to submit: qsub script.sh

#------ qsub options --------#
#$ -P qfe
#$ -M nmatsum@bu.edu
##### run time limit. format: hh:mm:ss; default 12 hrs
#$ -l h_rt=48:00:00
#$ -pe omp 12

# --------- job info -----------#

echo "start"
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "NSlots : $NSLOTS"
echo "Host name : $HOSTNAME"
echo "working directory : $TMPDIR"
echo "=========================================================="

#-------------#

module load gcc

#------- Program execution -------#

echo "running program"
pwd
./wolff.o
echo "finished"
