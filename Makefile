main: src/main.py
	python3 src/main.py --nsteps=100

mesh : src/mesh.py
	python3 src/mesh.py

continent : src/mesh_continent.py
	python3 src/mesh_continent.py 

altitude : altitude.py
	python3 altitude.py