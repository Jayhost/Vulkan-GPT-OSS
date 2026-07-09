# Compiler and Linker
CXX          = g++
CXXFLAGS     = -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter
LDFLAGS      = -lvulkan -lpthread -lm

# Shader Compiler (from Vulkan SDK)  -fsanitize=address
GLSLC        = glslangValidator

# Directories
SRC_DIR      = .
SHADER_DIR   = shaders
BUILD_DIR    = build

# Target executable
TARGET       = $(BUILD_DIR)/llm_vulkan

# C++ Sources
CPP_SOURCES  = main.cpp gguf.cpp model.cpp tokenizer.cpp vulkan_backend.cpp
CPP_OBJECTS  = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))

# GLSL Shader Sources
SHADER_SOURCES = \
    rmsnorm.comp \
    matvec_f32.comp \
    matvec_q4_0.comp \
    matvec_q4_0_expert.comp \
    rope.comp \
    attention.comp \
    swiglu.comp \
    add.comp \
    copy_to_kv.comp \
    matvec_f16.comp \
    matvec_f16_expert.comp \
    matvec_f32_expert.comp 
    

SHADER_OUTPUTS = $(patsubst %.comp,$(SHADER_DIR)/%.spv,$(SHADER_SOURCES))

# Default target
all: $(SHADER_DIR) $(BUILD_DIR) $(TARGET)

# Create directories
 $(SHADER_DIR):
	mkdir -p $(SHADER_DIR)

 $(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to compile .comp to .spv
 $(SHADER_DIR)/%.spv: $(SHADER_DIR)/%.comp
	$(GLSLC) -V $< -o $@

# Rule to compile C++ objects
 $(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the final executable
 $(TARGET): $(CPP_OBJECTS) $(SHADER_OUTPUTS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPP_OBJECTS) -o $(TARGET) $(LDFLAGS)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(SHADER_DIR)/*.spv

.PHONY: all clean