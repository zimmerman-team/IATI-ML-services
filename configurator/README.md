Configurator
============

The configurator is a package to allow easy configuration of systems that need to be deployed on different
machines.

Usage
-----

Just make sure that the `configurator/` directory is located in the root of your project.

Then you can import it as `from configurator import config`

The `config` module will look for a configuration file named `<hostname>.yml` in the `configs`
directory. Where `<hostname>` is the name of the machine where the program is being run.

All entries in the yml file will have automatically populated the `config` module.

Nested entries, will be accessible from the module as well. For example, this piece of configuration:

```
foo:
    - bar:
        - baz: xyz
```

will be accessible as `config.foo.bar.baz` ( == `"xyz"`).

