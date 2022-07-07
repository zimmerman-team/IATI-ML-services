# large_mp

`large_mp` is a library that extends Airflow's message passing features 
by allowing the transfer of big objects.
This is done by creation of temporary, randomly-named files that will
contain a dump of the serialized object.

This is done mainly by three functions:

## large_mp.send(ti,data)

Will JSON-serialize and store in a temporary file the object

## large_mp.recv(ti, task_id)

Will look for an object created by the given task_id and is
de-serializing it and returning it to the calling task.

## large_mp.clear_recv(ti, task_id)

Removes the message-passed object from the message-passing system.
It has to be run, especially at the end of the tasks to 
save disk space.
