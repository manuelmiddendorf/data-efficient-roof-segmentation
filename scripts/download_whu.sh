#!/bin/sh
set -eu

root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
raw_dir="$root/data/raw/whu"
archive="$raw_dir/whu_aerial_0.3m_raster.zip"
dataset_dir="$raw_dir/3. The cropped image tiles and raster labels"
url="https://gpcv.whu.edu.cn/data/3.%20The%20cropped%20aerial%20image%20tiles%20and%20raster%20labels.zip"
expected_bytes=5432920408

mkdir -p "$raw_dir"
current_bytes=0
if [ -f "$archive" ]; then
  current_bytes="$(wc -c < "$archive" | tr -d ' ')"
fi
if [ "$current_bytes" -gt "$expected_bytes" ]; then
  echo "Existing archive is larger than expected: $current_bytes bytes." >&2
  exit 1
fi
if [ "$current_bytes" -lt "$expected_bytes" ]; then
  curl --fail --location --continue-at - --output "$archive" "$url"
fi

actual_bytes="$(wc -c < "$archive" | tr -d ' ')"
if [ "$actual_bytes" -ne "$expected_bytes" ]; then
  echo "Unexpected archive size: $actual_bytes bytes (expected $expected_bytes)." >&2
  exit 1
fi

if [ ! -d "$dataset_dir" ]; then
  unzip -q "$archive" -d "$raw_dir"
fi

echo "WHU Aerial 0.3 m raster archive is available under data/raw/whu/."
