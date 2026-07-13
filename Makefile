VENV_DIR := venv
MODELS_DIR := models

.PHONY: all env install clone-repos setup clean

all: env clone-repos install setup

# Create virtual environment
env:
	python -m venv $(VENV_DIR)

# Clone required repositories, overwriting if they exist
clone-repos:
	@echo "Checking and cloning required repositories..."
	@mkdir -p $(MODELS_DIR)
	@if [ -d "$(MODELS_DIR)/d-mercator" ]; then \
		echo "Removing existing d-mercator..."; \
		rm -rf $(MODELS_DIR)/d-mercator; \
	fi
	@echo "Cloning d-mercator..."; \
	cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/d-mercator.git
	@if [ -d "$(MODELS_DIR)/hypermap" ]; then \
		echo "Removing existing hypermap..."; \
		rm -rf $(MODELS_DIR)/hypermap; \
	fi
	@echo "Cloning hypermap..."; \
	cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/hypermap.git
	@if [ -d "$(MODELS_DIR)/lorentz" ]; then \
		echo "Removing existing lorentz..."; \
		rm -rf $(MODELS_DIR)/lorentz; \
	fi
	@if [ -d "$(MODELS_DIR)/lorentz-embeddings" ]; then \
		echo "Removing existing lorentz..."; \
		rm -rf $(MODELS_DIR)/lorentz-embeddings; \
	fi
	@echo "Cloning lorentz-embeddings..."; \
	cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/lorentz-embeddings.git; \
	mv lorentz-embeddings lorentz; \
	rm -rf $(MODELS_DIR)/lorentz-embeddings
	@if [ -d "$(MODELS_DIR)/PoincareMaps" ]; then \
		echo "Removing existing PoincareMaps..."; \
		rm -rf $(MODELS_DIR)/PoincareMaps; \
	fi
	@echo "Cloning PoincareMaps..."; \
	cd $(MODELS_DIR) && git clone https://github.com/CicadaUY/PoincareMaps.git
	@echo "Repository cloning complete!"

# Activate env and install dependencies
install: 
	. $(VENV_DIR)/bin/activate && \
	pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt

# Run setup.py using the virtualenv
setup:
	. $(VENV_DIR)/bin/activate && \
	cd models/d-mercator/python && pip install -e . && \
	cd ../../../ && \
	$(VENV_DIR)/bin/python models/hypermap/python/setup.py build_ext --inplace

# Remove the virtual environment
clean:
	rm -rf $(VENV_DIR)
