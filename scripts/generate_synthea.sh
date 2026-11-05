#!/usr/bin/env bash
# Generate synthetic FHIR R4 patients with Synthea (requires Java 17+).
#   ./scripts/generate_synthea.sh [N_PATIENTS] [STATE]
set -euo pipefail

N="${1:-20}"
STATE="${2:-Massachusetts}"
DIR=".synthea"
JAR="$DIR/synthea-with-dependencies.jar"
URL="https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"

mkdir -p "$DIR"
[[ -f "$JAR" ]] || curl -fL -o "$JAR" "$URL"

# -s fixes the random seed so the dataset is reproducible across machines.
java -jar "$JAR" -s 42 -p "$N" "$STATE" \
  --exporter.baseDirectory "$DIR/output" \
  --exporter.fhir.export true \
  --exporter.fhir.transaction_bundle true \
  --exporter.hospital.fhir.export true \
  --exporter.practitioner.fhir.export true \
  --exporter.csv.export false

echo "Generated bundles in $DIR/output/fhir — load them with:"
echo "  uv run python scripts/load_fhir_bundles.py $DIR/output/fhir"
