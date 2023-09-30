from functools import wraps
import html
import time

from modules import shared, progress, errors, devices, fifo_lock

queue_lock = fifo_lock.FIFOLock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f




#MJ called from in ui.create_ui():  txt2img_args = dict(
                # fn=wrap_gradio_gpu_call(modules.txt2img.txt2img, extra_outputs=[None, '', '']),
                # _js="submit",
def wrap_gradio_gpu_call(func, extra_outputs=None):  #MJ: wrap_gradio_gpu_call() is the entry point of function execution in webui
    @wraps(func)
    def f(*args, **kwargs):

        # if the first argument is a string that says "task(...)", it is treated as a job id
        if args and type(args[0]) == str and args[0].startswith("task(") and args[0].endswith(")"):
            id_task = args[0]
            progress.add_task_to_queue(id_task)
        else:
            id_task = None

        with queue_lock:
            shared.state.begin(job=id_task)
            progress.start_task(id_task)

            try:
                res = func(*args, **kwargs) #MJ: func = modules.txt2img.txt2img,
                
                progress.record_results(id_task, res)
            finally:
                progress.finish_task(id_task)

            shared.state.end()

        return res
    #def f(*args, **kwargs)
    return wrap_gradio_call(f, extra_outputs=extra_outputs, add_stats=True) #MJ: wrap f with statistics of memory usages and reporting error messages

# Yes, you can directly call my_decorator(example); 
# this would apply the my_decorator to the example function and return the wrapped version of the example function. This is essentially what is happening under the hood when you use the @my_decorator syntax above the function definition; 
# it is syntactic sugar for example = my_decorator(example).

# def my_decorator(f):
#     @wraps(f)
#     def wrapper(*args, **kwds):
#         print('Calling decorated function')
#         return f(*args, **kwds)
#     return wrapper

# def example():
#     """Docstring for function example."""
#     print('Called example function')

# example = my_decorator(example)
# example()

# from functools import wraps

# def my_decorator(f):
#     @wraps(f)
#     def wrapper(*args, **kwds):
#         print('Calling decorated function')
#         return f(*args, **kwds)
#     return wrapper

# @my_decorator
# def example():
#     """Docstring for function example."""
#     print('Called example function')

# example()


def wrap_gradio_call(func, extra_outputs=None, add_stats=False): #MJ: called with add_stats = True; wrap_gradio_call() is the entry point of function execution in webui
    @wraps(func)
    #MJ: It is used to update the metadata of a decorated function to match the original function it is wrapping.
    # Essentially, it's used to carry the original function's name, module, and docstring to the wrapper function.
    #wrapper function = inner function
    #the metadata of the wrapper function f will be updated to reflect the metadata of func
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = shared.opts.memmon_poll_rate > 0 and not shared.mem_mon.disabled and add_stats
        if run_memmon:
            shared.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs)) #MJ: func = img2img
        except Exception as e: #MJ: [Tiled VAE]: Executing Encoder Task Queue: 100%|██████████████████████████████████████████| 364/364 [00:02<00:00, 153.69it/s][Tiled VAE]: Done in 5.251s, max VRAM alloc 23918.284 MB
            # When printing out our debug argument list,
            # do not print out more than a 100 KB of text
            max_debug_str_len = 131072
            message = "Error completing request"
            arg_str = f"Arguments of func txt2img or img2img: {args} {kwargs}"[:max_debug_str_len]
            if len(arg_str) > max_debug_str_len:
                arg_str += f" (Argument list truncated at {max_debug_str_len}/{len(arg_str)} characters)"
            errors.report(f"{message}\n{arg_str}", exc_info=True)

            shared.state.job = ""
            shared.state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            error_message = f'{type(e).__name__}: {e}'
            res = extra_outputs_array + [f"<div class='error'>{html.escape(error_message)}</div>"]

        devices.torch_gc()

        shared.state.skipped = False
        shared.state.interrupted = False
        shared.state.job_count = 0

        if not add_stats:
            return tuple(res)

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.1f} sec."
        if elapsed_m > 0:
            elapsed_text = f"{elapsed_m} min. "+elapsed_text

        if run_memmon: #MJ: mem_stats = {'min_free': 4061, 'free': 40787, 'total': 48677, 'active': 1, 'active_peak': 23919, 'reserved': 2622, 'reserved_peak': 39350, 'system_peak': 44617}
            mem_stats = {k: -(v//-(1024*1024)) for k, v in shared.mem_mon.stop().items()} #MJ: The expression -(v // -n) seems a bit unusual at first glance, but it's essentially a way to implement a "ceiling" division in Python.
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']  #MJ: min_free would represent the lowest value of free memory observed in any of those snapshots during the entire monitoring period.
            sys_pct = sys_peak/max(sys_total, 1) * 100

            toltip_a = "Active: peak amount of video memory used during generation (excluding cached data)"
            toltip_r = "Reserved: total amout of video memory allocated by the Torch library "
            toltip_sys = "System: peak amout of video memory allocated by all running programs, out of total capacity"

            text_a = f"<abbr title='{toltip_a}'>A</abbr>: <span class='measurement'>{active_peak/1024:.2f} GB</span>"
            text_r = f"<abbr title='{toltip_r}'>R</abbr>: <span class='measurement'>{reserved_peak/1024:.2f} GB</span>"
            text_sys = f"<abbr title='{toltip_sys}'>Sys</abbr>: <span class='measurement'>{sys_peak/1024:.1f}/{sys_total/1024:g} GB</span> ({sys_pct:.1f}%)"

            vram_html = f"<p class='vram'>{text_a}, <wbr>{text_r}, <wbr>{text_sys}</p>"
        else:
            vram_html = ''

        # last item is always HTML
        # outputs=[
        #             img2img_gallery,
        #             generation_info,
        #             html_info,
        #             html_log,
        #         ],
        
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr><span class='measurement'>{elapsed_text}</span></p>{vram_html}</div>"

        return tuple(res)
        #def f(*args, extra_outputs_array=extra_outputs, **kwargs)
    return f
