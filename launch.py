#MJ: To invoke the interactive vscode debugger:
#https://code.visualstudio.com/docs/python/debugging
import debugpy

# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# debugpy.breakpoint()
# print('break on this line')



from modules import launch_utils

args = launch_utils.args
python = launch_utils.python #MJ: = '/home/moon/stable-diffusion-webui/venv/bin/python'
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos #MJ: = 'repositories'

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive

list_extensions = launch_utils.list_extensions
#MJ: => def list_extensions(settings_file)

run_extension_installer = launch_utils.run_extension_installer
#MJ: => def run_extension_installer(extension_dir)

prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start


def main():
    #MJ: To use the following line for debugging, run python launch.py on the terminal and 
    # debugpy.listen(5678)
    # print("Wainting for debugger attach")


    # debugpy.wait_for_attach()
    # debugpy.breatpoint()
    # print("break on this line")


    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    if args.test_server:
        configure_for_tests()

    start()
    
    # def start():
    # print(f"Launching {'API server' if '--nowebui' in sys.argv else 'Web UI'} with arguments: {' '.join(sys.argv[1:])}")
    # import webui
    # if '--nowebui' in sys.argv:
    #     webui.api_only()
    # else:
    #     webui.webui()




if __name__ == "__main__":   
    main() #MJ: python launch.py ==> The code is initiated by launching the main module
