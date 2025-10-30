python3 benchmarks.py --U u1 --D double_lim_J3J2
python3 benchmarks.py --U u2 --D double_lim_J3J2 --gamma 0.001
python3 benchmarks.py --U symm --D double_lim_J3J2

python3 benchmarks.py --U u1 --D single_lim_disc_t
python3 benchmarks.py --U u2 --D single_lim_disc_t --gamma 0.001
python3 benchmarks.py --U symm --D single_lim_disc_t

python3 benchmarks.py --U u1 --D single_J3
python3 benchmarks.py --U u2 --D single_J3 --gamma 0.001
python3 benchmarks.py --U symm --D single_J3
