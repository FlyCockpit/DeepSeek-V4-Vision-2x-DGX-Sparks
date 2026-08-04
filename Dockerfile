# Thin layer over the Anemll Spark vLLM runtime: install the vision plugin
# EDITABLE so the bind-mounted source at /opt/dsv4-vision-plugin is what
# actually executes. Iterating on the plugin needs only a restart, never a rebuild.
FROM ghcr.io/anemll/dspark-vllm-gx10:0.1.1

COPY plugin /opt/dsv4-vision-plugin
RUN python3 -m pip install -e /opt/dsv4-vision-plugin --no-deps
