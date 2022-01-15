# git clone https://github.com/ruomingp/jax_playground.git && sh jax_playground/setup-tpu.sh
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r jax_playground/requirements-tpu.txt
git clone https://github.com/google/jax
pip install -e jax/
python3 jax_playground/pjit_test.py
