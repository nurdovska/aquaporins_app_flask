from flask import Flask, render_template, request, session, redirect, render_template_string, make_response
import lib as lib
app = Flask(__name__)
import uuid

@app.route("/")
def index():

	models_list = [['Setting1', 'Model 1 - trained on all 17 AQP types'], ['Setting2', 'Model 2 - trained on 10 AQP types'], ['Setting3', 'Model 3 - trained on mammalian AQPs only'], ['Setting4', 'Model 4 - trained on classical AQPs only']]
	return render_template("index.html", models_list=models_list)

@app.route("/loading", methods = ['POST', 'GET'])
def loading():
	return render_template("loading.html", setting=request.form["setting"], seq=request.form["sequence"])

@app.route("/results", methods = ['POST', 'GET'])
def results():
	print('args:', request.args)
	print('form:', request.form)

	seq = request.args['seq'].replace(" ", "").replace("%20", "")
	setting = request.args['setting']

	if ((len(seq)<180) or (lib.validate_protein_sequence(seq) == False) or (len(seq)>1500)):
		return render_template("results_page_error.html")
	else:
		model = lib.load_models('models/'+setting+'/best_model.h5')
		tokenizer_encoder, tokenizer_decoder = lib.get_tokenizer_encoder_decoder('models/dataset.csv')

		#get the prediction
		prediction = lib.predict_sequence(seq, model, tokenizer_encoder,tokenizer_decoder)


		#get the helical parts
		s = lib.print_helix_location(lib.classes_to_secondary_structure(prediction))
		helix_names = s[0]
		helix_locations = s[1]

		#the prediction as a dataframe
		df = lib.visualize_dataframe_prediction(seq, prediction)

		#get the plot
		random_string = uuid.uuid4().hex
		path = "static/" + random_string + ".png"
		lib.create_plot(seq, prediction, path, "P-I", "L-I", "L-S", "P-S", ['darkorange', 'cornflowerblue', 'plum', 'limegreen'])

		return render_template("results_page.html", Prediction = str(prediction), helix_names = helix_names, helix_locations = helix_locations, column_names=df.columns.values, row_data=list(df.values.tolist()), zip = zip, plot = path)



