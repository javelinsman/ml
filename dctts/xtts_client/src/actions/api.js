import fetch from 'cross-fetch'

export const baseURL = 'http://147.46.215.181:6007/'

export function fetchExperimentNames() {
    return fetch(`${baseURL}experiment_names`).then(res => res.json())
}

export function fetchTags(experimentName) {
    return fetch(`${baseURL}tags?experiment_name=${experimentName}`).then(res => res.json())
}

export function fetchSteps(experimentName) {
    return fetch(`${baseURL}steps?experiment_name=${experimentName}`).then(res => res.json())
}

export function fetchTensors(experimentName, tag, step) {
    return fetch(`${baseURL}tensors?experiment_name=${experimentName}&tag=${tag}&step=${step}`).then(res => res.json())
}