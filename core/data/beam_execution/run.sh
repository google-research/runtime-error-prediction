python3 -m \
    core.data.beam_execution.runner \
    --region us-central1 \
    --input gs://dataflow-samples/shakespeare/kinglear.txt \
    --output gs://project-codenet-trace-collection/results/output \
    --runner DataflowRunner \
    --project runtime-error-problems \
    --temp_location gs://project-codenet-trace-collection/temp/
