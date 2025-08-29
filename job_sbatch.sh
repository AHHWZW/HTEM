#!/bin/bash
# File: slurm_retry_manager.sh
# function: Submit tasks, monitor, error handling, and auto-retry


# --------------------------------------------
# Analyze errors based on the log file
# --------------------------------------------
analyze_errors() {
  current_dir=$(pwd)
  target_dir=$1

  out_log=logs/$2.out
  error_log=logs/$2.err
  new_params=""
  if grep -q -e "KILLED BY SIGNAL: 9" -e "OUT OF MEMORY" "$out_log" || grep -q -e "KILLED BY SIGNAL: 9" -e "OUT OF MEMORY" "$error_log"  ; then
    old_KPAR=$(grep "KPAR" INCAR | awk '{print $3}')
    if [ $(($old_KPAR % 2)) -eq 0 ]; then
      new_KPAR=$(($old_KPAR / 2))
    elif [ $(($old_KPAR % 3)) -eq 0 ]; then
      new_KPAR=$(($old_KPAR / 3))
    elif [ $(($old_KPAR % 5)) -eq 0 ]; then
      new_KPAR=$(($old_KPAR / 5))
    else
      new_KPAR=1
    fi
    new_params="${new_params} KPAR=$new_KPAR"
    sed -i "s/KPAR = $old_KPAR/KPAR  = $new_KPAR/" INCAR
  fi

  # Check if the electronic steps are not converging
  if grep -q "EDIFF" $error_log; then
    ediff=$(grep "EDIFF" INCAR | awk '{print $3}')
    new_ediff=$(echo "$ediff * 0.1" | bc)
    new_params="${new_params} EDIFF=$new_ediff"
  fi
  
  # Check if the ionic steps are broken
  if grep -q "ZBRENT" $error_log ; then
    if grep -q "POTIM" INCAR; then
      old_potim=$(grep "POTIM" INCAR | awk '{print $3}')
      new_potim=$(echo "$old_potim * 0.5" | bc)
      sed -i "s/POTIM = $old_potim/POTIM = $new_potim/" INCAR
      new_params="${new_params} POTIM=$new_potim"
    fi
    #new_params="${new_params} IBRION=1"
  fi

  echo "$new_params"
}

analysis_TIME_OUT() {
  current_dir=$(pwd)
  cd $target_dir
  new_params=""
  current_step=$(grep "ISTEP" OUTCAR | tail -n1 | awk '{print $3}')
  max_steps=$(grep "NSW" INCAR | awk '{print $3}')
  if [ -z $max_steps ]; then
    max_steps=60
  fi
  ratio=$(echo "scale=2; $max_steps/$current_steps" | bc)
  # Get the max time from the job script
  script_time=$(grep "SBATCH --time=" $JOB_SCRIPT | awk -F'=' '{print $2}')
  # Convert to minutes
  max_time=$(echo $script_time | awk -F':' '{print $1 * 60 + $2}')
  if [ $current_step -lt $max_steps ] && [ $max_time -lt 2880 ]; then
    new_max_time=$(int($ratio*$max_time))
    # Convert to HH:MM:SS format
    new_max_time=$(echo $new_max_time | awk '{print int($1/60)":"int($1%60)":"0}')
    new_params="time=$new_max_time"
    sed -i "s/^ --time=$script_time/ --time=$new_max_time/" $JOB_SCRIPT
  else
    new_params="None"
  fi
  cd $current_dir
  echo "$new_params"
}

relax_check() {
  current_dir=$(pwd)
  target_dir=$1
  cd $target_dir
  out_log=logs/$2.out
  new_params=""

  if grep -q "reached required accuracy" $out_log; then
    echo "REACHED|None"
  else
    g_F=$(grep "g(F)" $out_log | tail -1 | awk '{print -1*$5}')
    old_EDIFFG=$(grep "EDIFFG" INCAR | awk '{print $3}')

    if awk -v limit=-0.01 -v ediffg="$g_F" 'BEGIN { exit (ediffg > limit && ediffg < 0) ? 0 : 1 }'; then
      new_EDIFFG=$(echo "$old_EDIFFG * 5" | bc)
      new_params="EDIFFG=$new_EDIFFG"
      sed -i "s/EDIFFG = $old_EDIFFG/EDIFFG = $new_EDIFFG/" INCAR
      echo "RELAX_RETRY|$new_params"  
    else
      echo "RELAX_FAILED|None"
    fi
  fi
  cd $current_dir

}

# --------------------------------------------
# Step 1: Submit initial tasks and record metadata
# --------------------------------------------
JOB_SCRIPT="HTEM_slurm_sub.sh"      # The job script to be submitted
MAX_RETRIES=2                 # The maximum number of retries
PARAMS="INCAR_PARAMS"         # The parameters to be passed to the job script
RETRY_DB="jobs_db.csv"       # database of jobs
CHECK_INTERVAL=15              # check_interval (s)
# Read command line arguments, optional modes: relax1, strain, NPT
if [ $# -eq 0 ]; then
  echo "Usage: $0 [relax1|relax2|static|NVT|NPT]"
  exit 1
fi

JOB_SCRIPT_dir=$(which $JOB_SCRIPT)
cp $JOB_SCRIPT_dir $JOB_SCRIPT
# Read the command line argument
MODE=$1

check_jobs() {
  if grep -q -E 'PENDING' "$RETRY_DB"; then
    return 0  # Remaining jobs
  else
    return 1  # All jobs completed
  fi
}

# Submit the initial tasks to SLURM
echo "Submitting initial tasks..."
job_ids=()

if [ "$MODE" = "relax1" ]; then
  for i in $(ls -lv | grep '^d' | awk '{print $NF}' | grep '^state_')
  do cd $i;  
    pwd
    if [ ! -d "logs" ]; then
      mkdir logs
    fi
    cp ../HTEM_slurm_sub.sh .
    JOB_ID=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
    echo "$i,relax,0,0,$JOB_ID,PENDING,$PARAMS,0" >> ../$RETRY_DB  
    cd ..
  done
elif [ "$MODE" = "NPT" ]; then
  for i in $(ls -lv | grep '^d' | awk '{print $NF}' | grep '^state_')
  do cd $i/NPT;  
    pwd
    if [ ! -d "logs" ]; then
      mkdir logs
    fi
    cp ../../HTEM_slurm_sub.sh .
    JOB_ID=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
    echo "$i,NPT,0,0,$JOB_ID,PENDING,$PARAMS,0" >> ../../$RETRY_DB 
    cd ../../
  done
elif [ "$MODE" = "relax2" ] || [ "$MODE" = "static" ] || [ "$MODE" = "NVT" ]; then
  for i in $(ls -lv | grep '^d' | awk '{print $NF}' | grep '^state_')
  do cd $i;
    for j in $(ls -lv | grep '^d' | awk '{print $NF}' | grep '^Dst_')
    do cd $j;
      for k in $(ls -lv | grep '^d' | awk '{print $NF}')
      do cd $k;
        pwd
        if [ ! -d "logs" ]; then
          mkdir logs
        fi
        cp ../../../HTEM_slurm_sub.sh .
        JOB_ID=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
        echo "$i,strain,$j,$k,$JOB_ID,PENDING,$PARAMS,0" >> ../../../$RETRY_DB 
        cd ../
      done
      cd ../
    done
    cd ../
  done
else
  echo "Usage: $0 [relax1|relax2|static|NVT|NPT]"
  exit 1
fi

# --------------------------------------------
# Step 2: Monitor and retry tasks 
# --------------------------------------------
# Start to monitor the pending tasks
while check_jobs; do
  echo "-------------------------------------"
  echo "[$(date)] checking unfinished jobs..."
  
  # Get all pending jobs
  pending_jobs=$(awk -F',' '$6=="PENDING"{print $5}' $RETRY_DB)
  
  for job_id in $pending_jobs; do
    job_state=$(sacct -j $job_id --format=State --noheader | head -n1 | tr -d ' ')
    
    case $job_state in
      "COMPLETED")
        case $MODE in 
          "relax1"|"relax2")
            # Read the old parameters and update the job scrip
            IFS=',' read -r -a job_info <<< $(awk -F',' -v id="$job_id" '$5==id {print $1","$2","$3","$4","$5","$6","$7","$8}' $RETRY_DB)
            
            if [ "${job_info[1]}" = "strain" ]; then
              target_dir=${job_info[0]}/${job_info[2]}/${job_info[3]}
            else
              target_dir=${job_info[0]}
            fi
            target_dir=${job_info[0]}
            result=$(relax_check "$target_dir" "$job_id")
            status=$(echo "$result" | cut -d'|' -f1)
            new_params=$(echo "$result" | cut -d'|' -f2)
       
            if [ "$status" = "REACHED" ]; then
              echo "Task $job_id completed."
              sed -i "s/,$job_id,PENDING/,$job_id,COMPLETED/" $RETRY_DB
            elif [ "$status" = "RELAX_RETRY" ]; then
              echo "Relaxation for task $job_id requires retry with new parameters: $new_params."
              current_retry=${job_info[7]}
              if [ $current_retry -lt $MAX_RETRIES ]; then
                current_retry=$(($current_retry+1))
                current_dir=$(pwd)
                cd $target_dir
                new_job_id=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
                echo "Resubmitted job $new_job_id (retry: $((current_retry)))."
                cd $current_dir
                sed -i "s/,$job_id,PENDING/,$job_id,RETRYING/" $RETRY_DB
                echo "${job_info[0]},${job_info[1]},${job_info[2]},${job_info[3]},$new_job_id,PENDING,\"$new_params\",$((current_retry))" >> $RETRY_DB
              else
                echo "Task $job_id reached max retries."
                sed -i "s/,$job_id,PENDING/,$job_id,FAILED/" $RETRY_DB
              fi
            else
              echo "Relaxation for task $job_id failed."
              sed -i "s/,$job_id,PENDING/,$job_id,FAILED/" $RETRY_DB
            fi
            ;;
          "static"|"NVT"|"NPT")
            echo "Task $job_id completed."
            sed -i "s/,$job_id,PENDING/,$job_id,COMPLETED/" $RETRY_DB
        esac

        ;; 
      "PENDING")
        echo "Task $job_id is still pending."
        ;;

      "RUNNING")
        echo "Task $job_id is still running."
        ;;
        
      "FAILED"|"TIMEOUT"|"NODE_FAIL"|"CANCELLED"*)
        echo "Task $job_id failed: $job_state"
        current_retry=$(awk -F',' -v id="$job_id" '$5==id {print $8}' $RETRY_DB)
        
        if [ $current_retry -ge $MAX_RETRIES ]; then
          echo "Task $job_id reached max retries."
          sed -i "s/,$job_id,PENDING/,$job_id,FAILED/" $RETRY_DB
        else
          current_retry=$(($current_retry+1))
          # Read the old parameters and update the job script
          IFS=',' read -r -a job_info <<< $(awk -F',' -v id="$job_id" '$5==id {print $1","$2","$3","$4","$5","$6","$7","$8}' $RETRY_DB)
          # Retry the job
          case $MODE in
            "relax1")
              target_dir=${job_info[0]}
              cd $target_dir
              new_params=$(analyze_errors "$target_dir" "$job_id")
              new_job_id=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
              sed -i "s/,$job_id,PENDING/,$job_id,RETRYING/" ../$RETRY_DB
              cd ../
              ;;
            "NPT")
              target_dir=${job_info[0]}/NPT
              cd $target_dir
              new_params=$(analyze_errors "$target_dir" "$job_id")
              new_job_id=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
              sed -i "s/,$job_id,PENDING/,$job_id,RETRYING/" ../../$RETRY_DB
              cd ../../
              ;;
            "relax2"|"static"|"NVT")
              target_dir=${job_info[0]}/${job_info[2]}/${job_info[3]}
              cd $target_dir
              new_params=$(analyze_errors "$target_dir" "$job_id")
              new_job_id=$(sbatch --parsable --output=logs/%j.out --error=logs/%j.err $JOB_SCRIPT)
              sed -i "s/,$job_id,PENDING/,$job_id,RETRYING/" ../../../$RETRY_DB
              cd ../../../
              ;;
          esac
          
          # Update the retry database
          #sed -i "s/,$job_id,PENDING/,$job_id,RETRYING/" $RETRY_DB
          echo "${job_info[0]},${job_info[1]},${job_info[2]},${job_info[3]},$new_job_id,PENDING,\"$new_params\",$((current_retry))" >> $RETRY_DB
          echo "Resubmitted job $new_job_id (retry: $((current_retry)))"
        fi
        ;;
    esac
  done
  
  sleep $CHECK_INTERVAL
done

echo "All Tasks Completed!"
exit 0
