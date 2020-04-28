import React from 'react'
import { Slider, Typography } from '@material-ui/core'

const StepSlider = ({ steps, step, setStep }) => {
    const marks = steps.map((step, i) => ({value: i, label: step}))
    const valueText = value => steps[value]
    return (
        <div>
            <Typography id="discrete-slider-custom" gutterBottom>
                Step
            </Typography>
            <Slider
                defaultValue={steps.length - 1}
                getAriaValueText={valueText}
                aria-labelledby="discrete-slider-custom"
                step={1}
                min={0}
                max={steps.length - 1}
                marks={marks}
                onChange={(e, v) => setStep(steps[v])}
            />
        </div>
    )
}

export default StepSlider