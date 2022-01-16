# git clone https://github.com/ruomingp/jax_playground.git && sh jax_playground/setup-tpu.sh
sudo pip install -r jax_playground/requirements-tpu.txt
git clone https://github.com/google/jax
cd jax/
sudo pip install -e .[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd ..
python3 jax_playground/pjit_test.py
