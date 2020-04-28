import React from 'react'
import Tensor2dView from './Tensor2dView'
import { Card } from '@material-ui/core'

const TagView = ({ experimentName, tag, step }) => {
    return (
        <Card>
            <h6>{tag}</h6>
            <Tensor2dView experimentName={experimentName} tag={tag} step={step} />
        </Card>
    )
}

export default TagView