
# Peatland ABM Demo

Conceptual agent-based model built in Python 

## Installation & Running


1. Clone the repository:
	```bash
	git clone https://github.com/Geambrosio/peatland_abm.git
	cd peatland_abm
	```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

3. (Optional) Generate input data:
	You can generate a new agent profit input file by running:
	```bash
	python make_profits_csv.py
	```
	This will create `profits_agents.csv` with randomized agent profit data for use in the model.

4. Run the app:
	```bash
	streamlit run app.py
	```
	The model runs in an interactive web-based user interface (UI) in your browser. You can set parameters, run simulations, and view results visually.

## Requirements

All required Python libraries are listed in `requirements.txt`.

## License

Apache 2.0
