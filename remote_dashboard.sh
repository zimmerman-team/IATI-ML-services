#!/bin/bash

VM_URI=$(bash config/vm_uri.sh)

ssh -t $VM_URI "cd learning_sets ; ./system_dashboard.sh"