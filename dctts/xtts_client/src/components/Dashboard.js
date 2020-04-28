import React, {useState, useEffect} from 'react'
import { fetchExperimentNames, fetchSteps, fetchTags } from '../actions/api'
import Container from '@material-ui/core/Container'
import Grid from '@material-ui/core/Grid';
import TagView from './TagView'
import { Paper } from '@material-ui/core';
import NavBar from './NavBar';
import StepSlider from './StepSlider';

const ExperimentView = ({ experimentName }) => {
    const [tags, setTags] = useState([])
    const [steps, setSteps] = useState([])
    const [step, setStep] = useState(null)

    useEffect(() => {
        (async () => {
            setTags(await fetchTags(experimentName))
            setSteps(await fetchSteps(experimentName))
        })()
    }, [experimentName])
    useEffect(() => setStep(steps[steps.length - 1]), [steps])
    
    if (tags.length && steps.length && step) {
        return (
            <Grid container>
                <NavBar />
                <Grid item xs={12}>
                    <StepSlider steps={steps} step={step} setStep={setStep} />
                </Grid>
                <Grid item xs={12}>
                    <Paper>
                        <Grid container spacing={2}>
                            {tags.map((tag, i) => (
                                <Grid item xs={4} key={i}>
                                    <TagView experimentName={experimentName} tag={tag} step={step} />
                                </Grid>
                                )
                            )}
                        </Grid>
                    </Paper>
                </Grid>
            </Grid>
        )
    }
    return null
}

const Dashboard = () => {
    const [experimentNames, setExperimentNames] = useState([])
    useEffect(() => {
        (async () => {
            setExperimentNames(await fetchExperimentNames())
        })()
    }, [])
    return (
        <Container maxWidth="xl">
            {experimentNames.map((name, i) => <ExperimentView key={i} experimentName={name} />)}
        </Container>
    )
}

export default Dashboard;