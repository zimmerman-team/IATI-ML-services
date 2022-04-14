

To be run in this directory: 
    
    docker-compose up -d

What it does is it builds and runs all docker instances and is put in the background.
Omit the `-d` option to get it interactive

To stop everything run:

    docker-compose down

Advantages of dockerization:
- no need to specify specific listening hostnames in order to restrict access
- doesn't mess with hosting machine's installation and libraries
- replicable install of the services
- easy tear down of a service
- existing ecosystem to monitor running docker instances
- easy migration on a different machine
- straightforward multiple instances of a service. Useful for load balancing (kubernetes..)