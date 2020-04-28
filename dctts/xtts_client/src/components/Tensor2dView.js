import React, {useState, useEffect, useRef} from 'react'
import * as d3 from 'd3'
import { fetchTensors } from '../actions/api'

const Tensor2dView = ({ experimentName, tag, step }) => {
    const [tensors, setTensors] = useState([])
    const ref = useRef(null)
    useEffect(() => {
        if (ref.current) {
            (async () => {
                setTensors(await fetchTensors(experimentName, tag, step))
            })()
        }
    }, [experimentName, tag, step])
    useEffect(() => {
        if (tensors.length && ref.current) {
            const tensor = tensors[0]
            const svg = d3.select(ref.current)
            svg.selectAll('*').remove()
            const svgMaxWidth = 1000, svgMaxHeight=5000
            const [n, m] = [tensor.tensor.length, tensor.tensor[0].length]
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

export default Tensor2dView