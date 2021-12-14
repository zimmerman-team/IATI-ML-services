import line_profiler

profiler = line_profiler.LineProfiler()

def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)

    print(f"profiling function {func.__name__}")
    return inner
