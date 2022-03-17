#!/bin/bash -i
(
echo "starting dashbord for monitoring.."
sleep 2
tmux new-window "watch -t 'df -h'" \; \
  split-window -f -v "watch -t 'free -h'" \; \
  split-window -f -v "watch -t 'screen -ls'" \; \
  split-window -v "watch -t 'ps aux | grep airflow'" \; \
  split-window -f -h "watch -n 30 -t 'bash mongodb_diagnostics.sh'" \; \
  split-window -v "watch -n 10 -t 'python3 airflow/print_running_tasks.py'" \; \
  split-window -f -v "htop" \; \
  split-window -h "watch -n 600 -t 'python3 npas_splits_summary.py'" \;
) &
tmux
