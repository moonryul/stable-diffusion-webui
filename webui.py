from __future__ import annotations

import os
import time

from modules import timer
from modules import initialize_util
from modules import initialize

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.imports() #MJ: import modules and initialization of variables, and starting of memory thread; Modules execute statements

initialize.check_versions() 


def create_api(app): #MJ: app = FastAPI() 
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only():
    from fastapi import FastAPI
    from modules.shared_cmd_options import cmd_opts

    initialize.initialize() #MJ: => load the scripts etc.

    app = FastAPI() 
    initialize_util.setup_middleware(app)
    api = create_api(app)  #MJ:  api = Api(app, queue_lock)

    from modules import script_callbacks
    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    
    #MJ: api.launch() is implemented by uvicorn.run() as follows
    
    #  def launch(self, server_name, port, root_path):
    #     self.app.include_router(self.router)
    #     uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive, root_path=root_path)
      
    #MJ: in place of  app, local_url, share_url = shared.demo.launch() in webui()     
    api.launch(
        server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1",
        port=cmd_opts.port if cmd_opts.port else 7861,
        root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else ""
    )
#def api_only()

def webui():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api #MJ: false
    initialize.initialize()   #MJ: => load the scripts, load_model() [LatentDiffusion model and checkpoints]

    from modules import shared, ui_tempdir, script_callbacks, ui, progress, ui_extra_networks

    while 1: #MJ: The infinite interactive loop:
        
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()  #MJ: The cpu's control is jumped to here of the main thread while the "load_model" thread was running
        
        #MJ: ui.create_ui() returns demo defined by  with gr.Blocks(theme=shared.gradio_theme, analytics_enabled=False, title="Stable Diffusion") as demo:
        
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv('SD_WEBUI_RESTARTING') != '1':
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not any([cmd_opts.listen, cmd_opts.share, cmd_opts.ngrok, cmd_opts.server_name])



    
        #MJ: Launches a simple web server that serves the demo: Blocks, interacting with the user's activity of image generation
        #  app: FastAPI app object that is running the demo
        #  local_url: Locally accessible link to the demo
        #  share_url: Publicly accessible link to the demo (if share=True, otherwise None)
        app, local_url, share_url = shared.demo.launch( #MJ: shared.sd_model = LatentDiffusion(..) is not yet created
            share=cmd_opts.share,
            server_name=initialize_util.gradio_server_name(),
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=gradio_auth_creds,
            inbrowser=auto_launch_browser,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path,
            app_kwargs={
                "docs_url": "/docs",
                "redoc_url": "/redoc",
            },
            root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
        )

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        initialize_util.setup_middleware(app)

        progress.setup_progress_api(app)
        ui.setup_ui_api(app)

        if launch_api: #MJ: this is false; is true when "--api" commandline argument is passed
            create_api(app) #MJ: though there is no return, app object itself is modified.

        ui_extra_networks.add_pages_to_demo(app)

        startup_timer.record("add APIs")

        with startup_timer.subcategory("app_started_callback"):
            script_callbacks.app_started_callback(shared.demo, app)

        timer.startup_record = startup_timer.dump()
        print(f"Startup time: {startup_timer.summary()}.")

        try:
            while True: #MJ: infinite loop while waiting for the user input's "stop" or "restart" interrupt;
                #This process interleaves with the web server launched above
                
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
            #while True: #MJ: infinite loop            
                        
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break #MJ break out of the interactive infinite loop

        # disable auto launch webui in browser for subsequent UI Reload
        os.environ.setdefault('SD_WEBUI_RESTARTING', '1')

        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        
        initialize.initialize_rest(reload_script_modules=True)
    #while 1: #MJ: The infinite interactive loop

if __name__ == "__main__":
    from modules.shared_cmd_options import cmd_opts

    if cmd_opts.nowebui:
        api_only()
    else:
        webui()
