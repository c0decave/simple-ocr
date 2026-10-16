simple-ocr __VERSION__

Needs podman or docker, nothing else.

  cp .env.example .env      # set BACKEND and the model server's host, port and API key
  ./start.sh                # loads image.tar once, then serves http://127.0.0.1:8080

In the background, restarted after a reboot:

  podman load -i image.tar
  podman compose up -d      # or: docker load -i image.tar && docker compose up -d

The package holds only the program: no settings, no saved benchmark runs, no images.
Saved runs are kept in the volume simple-ocr-data on this machine.
COMMIT names the git commit the image was built from.
