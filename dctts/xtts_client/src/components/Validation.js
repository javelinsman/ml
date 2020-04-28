import React, {useState, useEffect, useRef} from 'react'
import fetch from 'cross-fetch'
import * as d3 from 'd3'

const Tensor2dView = ({ experimentName, tag, step }) => {
    const baseURL = 'http://147.46.215.181:6007/'
    const [tensors, setTensors] = useState([])
    const ref = useRef(null)
    useEffect(() => {
        if (ref.current) {
            (async () => {
                setTensors(await fetch(`${baseURL}tensors?experiment_name=${experimentName}&tag=${tag}&step=${step}`).then(res => res.json()))

            })()
        }
    }, [experimentName, tag, step])
    useEffect(() => {
        if (tensors.length && ref.current) {
            const tensor = tensors[0]
            const svg = d3.select(ref.current)
            const svgMaxWidth = 1000, svgMaxHeight=5000
            const [n, m] = [tensor.tensor.length, tensor.tensor[0].length]
            console.log({n, m})
            const rectMaxWidth = svgMaxWidth / m, rectMaxHeight = svgMaxHeight / n
            const w = Math.min(rectMaxWidth, rectMaxHeight)

            const colorScale = d3.scaleLinear().domain(d3.extent(tensor.tensor.reduce((a, b) => a.concat(b)))).range([0, 1])
            const color = d => d3.interpolateViridis(colorScale(d))

            const rows = svg.selectAll('g').data(tensor.tensor).enter().append('g')
                .attr('transform', (d, i) => `translate(${0},${i * w})`)
            rows.selectAll('rect').data(d => d).enter().append('rect')
                .attr('x', (d, i) => i * w).attr('y', 0).attr('width', w).attr('height', w)
                .style('fill', d => color(d)).style('stroke-width', 0)

        }
    }, [tensors, ref])
    return (
        <svg ref={ref}></svg>
    )
}


const TagView = ({ experimentName, tag }) => {
    const baseURL = 'http://147.46.215.181:6007/'
    const [allSteps, setAllSteps] = useState([])
    useEffect(() => {
        (async () => {
            setAllSteps(await fetch(`${baseURL}steps?experiment_name=${experimentName}&tag=${tag}`).then(res => res.json()))
        })()
    }, [experimentName, tag])
    if (allSteps.length) {
        const step = allSteps[0]
        return (
            <div>
                <h6>{tag}</h6>
                <Tensor2dView experimentName={experimentName} tag={tag} step={step} />
            </div>
        )
    }
    return null
}

const ExperimentView = ({ name }) => {
    const baseURL = 'http://147.46.215.181:6007/'
    const [tags, setTags] = useState([])
    useEffect(() => {
        (async () => {
            setTags(await fetch(`${baseURL}tags?experiment_name=${name}`).then(res => res.json()))
        })()
    }, [name])
    return (
        <div style={{display: 'flex'}}>
            {tags.map((tag, i) => <TagView key={i} experimentName={name} tag={tag} />)}
        </div>
    )
}

const Validation = () => {
    const baseURL = 'http://147.46.215.181:6007/'
    const [experimentNames, setExperimentNames] = useState([])
    useEffect(() => {
        (async () => {
            setExperimentNames(await fetch(`${baseURL}experiment_names`).then(res => res.json()))
        })()
    }, [])
    return (
        <div>
            {experimentNames.map((name, i) => <ExperimentView key={i} name={name} />)}
        </div>
    )
}

export default Validation;