from flask import Flask, request, jsonify
from flask_cors import CORS
from xtts.experiment import Experiment
import argparse
import os

log_dir = None

app = Flask(__name__)
CORS(app)

def get_experiment_names():
    return [i for i in os.listdir(log_dir)
            if not i.startswith('.')]

@app.route('/experiment_names', methods=['GET'])
def experiment_names():
    return jsonify(get_experiment_names())

@app.route('/tags', methods=['GET'])
def tags():
    experiment_name = request.args['experiment_name']
    if experiment_name in get_experiment_names():
        experiment = Experiment(experiment_name, log_dir)
        return jsonify(experiment.fetch_tags())
    else:
        return jsonify({
            'error': f'Experiment named {experiment_name} does not exists'
        })

@app.route('/steps', methods=['GET'])
def steps():
    experiment_name = request.args['experiment_name']
    tag = request.args['tag'] if 'tag' in request.args else '%'
    if experiment_name in get_experiment_names():
        experiment = Experiment(experiment_name, log_dir)
        return jsonify(experiment.fetch_steps(tag))
    else:
        return jsonify({
            'error': f'Experiment named {experiment_name} does not exists'
        })

@app.route('/tensors', methods=['GET'])
def tensors():
    experiment_name = request.args['experiment_name']
    tag = request.args['tag'] if 'tag' in request.args else '%'
    step = request.args['step'] if 'step' in request.args else '%'
    if experiment_name in get_experiment_names():
        experiment = Experiment(experiment_name, log_dir)
        return jsonify(experiment.fetch_tensors(tag, step))
    else:
        return jsonify({
            'error': f'Experiment named {experiment_name} does not exists'
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', action='store_const', default='0.0.0.0', const=str)
    parser.add_argument('--port', action='store_const', default=6007, const=int)
    parser.add_argument('--log_dir', action='store_const', default='xtts_logs', const=str)
    args = parser.parse_args()
    log_dir = args.log_dir
    app.run(host=args.host, port=args.port, threaded=True)