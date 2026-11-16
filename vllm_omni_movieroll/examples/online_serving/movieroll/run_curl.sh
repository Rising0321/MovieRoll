#!/bin/bash
# MovieRoll through /v1/videos: an asynchronous three-event job (events go in extra_params), polled, then downloaded.
# A synchronous request is the same form posted to /v1/videos/sync; DELETE /v1/videos/{id} cancels a running job.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-movieroll_events.mp4}"
NUM_FRAMES="${NUM_FRAMES:-81}"

create_response=$(
  curl -sS -X POST "${BASE_URL}/v1/videos" -H "Accept: application/json" \
    -F "prompt=A young wizard practises levitation in a torchlit stone chamber." \
    -F "size=864x480" -F "num_frames=${NUM_FRAMES}" -F "seed=930" \
    -F 'extra_params={"events": ["A young wizard practises levitation in a torchlit stone chamber.", "A small green dragon flies in and lands on his shoulder.", "The dragon puffs smoke at a book, whose pages start turning by themselves."]}'
)
field() { python3 -c "import json, sys; print(json.load(sys.stdin).get('$1', ''))"; }   # no jq needed
video_id="$(echo "${create_response}" | field id)"
echo "Created video job ${video_id}"
while true; do
  status="$(curl -sS "${BASE_URL}/v1/videos/${video_id}" | field status)"
  case "${status}" in
    queued|in_progress) sleep 5 ;;
    completed) break ;;
    *) echo "job ${video_id}: ${status}"; exit 1 ;;
  esac
done
curl -sS -o "${OUTPUT_PATH}" "${BASE_URL}/v1/videos/${video_id}/content"
echo "Saved ${OUTPUT_PATH}"
