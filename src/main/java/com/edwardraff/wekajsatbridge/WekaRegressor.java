package com.edwardraff.wekajsatbridge;

/*
 * Copyright (C) 2014 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


import java.util.List;
import java.util.concurrent.ExecutorService;
import jsat.classifiers.DataPoint;
import jsat.exceptions.FailedToFitException;
import jsat.parameters.Parameter;
import jsat.parameters.Parameterized;
import jsat.regression.RegressionDataSet;
import weka.classifiers.Classifier;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.WeightedInstancesHandler;

/**
 * This class wraps a Weka Classifier into a JSAT regressor with the associated
 * behavior. <br>
 * <br>
 * Parameters are inferred directly from matching get/set methods from the given
 * Weka classifier, rather than using the {@link OptionHandler} interface. This 
 * is done because the options array returned may have empty values, and the 
 * option arrays tend to have uninformative names. 
 * 
 * @author Edward Raff
 */
public class WekaRegressor implements jsat.regression.Regressor, Parameterized
{
    /**
     * When a weka classifier attempts to classify an instance, the instance 
     * MUST belong to a dataset, or an exception will be thrown. So 
     */
    private Instances wekaDataSet;
    private Classifier wekaClassifier;

    public WekaRegressor(Classifier wekaClassifier)
    {
        if(!wekaClassifier.getCapabilities().handles(Capability.NUMERIC_CLASS))
            throw new IllegalArgumentException("The given Weka classifier (" + wekaClassifier.getClass().getSimpleName() + ") dosn't support regression tasks");
        this.wekaClassifier = wekaClassifier;
    }
    
    public WekaRegressor(WekaRegressor toCopy)
    {
        this.wekaClassifier = OtherUtils.serializationCopy(toCopy.wekaClassifier);
        if(toCopy.wekaDataSet != null)
            this.wekaDataSet = OtherUtils.serializationCopy(new Instances(toCopy.wekaDataSet, 0));
    }

    @Override
    public boolean supportsWeightedData()
    {
        return wekaClassifier instanceof WeightedInstancesHandler;
    }

    @Override
    public double regress(DataPoint data)
    {
        try
        {
            Instance instance = InstanceHandler.dataPointToInstance(data);
            instance.setDataset(wekaDataSet);
            return wekaClassifier.classifyInstance(instance);
        }
        catch (Exception ex)
        {
            return Double.NaN;
        }
    }

    @Override
    public void train(RegressionDataSet arg0, ExecutorService arg1)
    {
        train(arg0);
    }

    @Override
    public void train(RegressionDataSet dataSet)
    {
        try
        {
            Instances instances = InstanceHandler.dataSetToInstances(dataSet);
            this.wekaDataSet = OtherUtils.serializationCopy(new Instances(instances, 0));
            wekaClassifier.buildClassifier(instances);
        }
        catch (Exception ex)
        {
            throw new FailedToFitException(ex);
        }
    }

    @Override
    public WekaRegressor clone()
    {
        return new WekaRegressor(this);
    }
    
    @Override
    public List<Parameter> getParameters()
    {
        return Parameter.getParamsFromMethods(wekaClassifier);
    }

    @Override
    public Parameter getParameter(String paramName)
    {
        return Parameter.toParameterMap(getParameters()).get(paramName);
    }
}
