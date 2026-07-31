#!/bin/sh
# Regenerate the committed SPIR-V from the GLSL sources. Vulkan has no runtime
# GLSL compiler, so unlike the Metal shaders -- compiled from source by the
# driver at startup -- these ship as .spv artifacts with the sources beside
# them. Run after editing any .comp file; needs glslangValidator (apt:
# glslang-tools, brew: glslang).
set -e
cd "$(dirname "$0")"
for f in *.comp; do
    glslangValidator --target-env vulkan1.1 -o "${f%.comp}.spv" "$f"
    echo "built ${f%.comp}.spv"
done
