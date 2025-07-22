mkdir -p .streamlit
cat > .streamlit/setup.sh << 'EOF'
#!/usr/bin/env bash
# Synchronise la config des submodules
git submodule sync --recursive
# Récupère et initialise tous les submodules
git submodule update --init --recursive
EOF
chmod +x .streamlit/setup.sh
