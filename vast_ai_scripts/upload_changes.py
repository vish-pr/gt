import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os


class MyHandler(FileSystemEventHandler):

  def on_modified(self, event):
    if not event.is_directory and event.src_path.endswith('.py'):
      print(f'File {event.src_path} has been modified')
      self.upload_file(event.src_path)

  def upload_file(self, file_path):
    print('/'.join(file_path.split('/')[4:]))

    # Get the server path and port from environment variables
    server_path = os.getenv('SERVER')
    server_port = os.getenv('PORT')
    # export SERVER_PATH=/path/to/server
    # export SERVER_PORT=8000

    if server_path is None or server_port is None:
      print('Please set SERVER_PATH and SERVER_PORT environment variables', server_path, server_port)
      return
    target_server = server_path + ':/root/' + '/'.join(file_path.split('/')[4:])
    subprocess.run(['scp', '-P ' + server_port, file_path, target_server])
    print(f'{file_path} uploaded to {target_server}')


if __name__ == "__main__":
  observer = Observer()
  files_to_watch = ['/home/v/github/gt/bot/', '/home/v/github/gt/layers/', '/home/v/github/gt/hub/']
  for file_path in files_to_watch:
    observer.schedule(MyHandler(), file_path, recursive=True)
  observer.start()
  try:
      while True:
          time.sleep(5)
  except KeyboardInterrupt:
      observer.stop()
  observer.join()
