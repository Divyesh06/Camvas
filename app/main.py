import sys
import threading
import time

def status_callback(status):
    if status == "stopped":
        VirtualCam.stop_camera()

    else:
        import time
        time.sleep(1)
        threading.Thread(
            target=VirtualCam.main,
            kwargs={
                "started_callback": on_camera_started,
                "disconnected_callback": on_camera_disconnected,
            },
            daemon=True,
        ).start()

def on_camera_started():
    status_app_window.set_state("started")

def on_camera_disconnected():
    status_app_window.set_state("waiting")

def is_started_from_windows_startup():
    return '--startup' in sys.argv

def on_camera_stopped():
    VirtualCam.stop_camera()

if __name__ == "__main__":
    if not is_started_from_windows_startup():

        import loading_screen
        loading_screen.start()

        import VirtualCam
        import status_app
        status_app_window = status_app.CamvasStatusApp(callback=status_callback)
        loading_screen.stop()

        status_app_window.run()

    else:

        import VirtualCam
        import status_app

        status_app_window = status_app.CamvasStatusApp(callback=status_callback, silent=True)
        status_app_window.run()
