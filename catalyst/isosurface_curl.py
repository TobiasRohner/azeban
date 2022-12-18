from paraview.simple import *

from paraview.catalyst import get_args, get_execute_params
from paraview import catalyst

# print values for parameters passed via adaptor (note these don't change,
# and hence must be created as command line params)
print("executing catalyst_pipeline")
print("===================================")
print("pipeline args={}".format(get_args()))
print("execute params={}".format(get_execute_params()))
print("===================================")


# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")


view = CreateRenderView()

computeDerivatives1 = ComputeDerivatives(registrationName='ComputeDerivatives1', Input=producer)
computeDerivatives1.Scalars = ['POINTS', 'u']
computeDerivatives1.Vectors = ['POINTS', 'velocity']
computeDerivatives1.OutputVectorType = 'Vorticity'
computeDerivatives1.OutputTensorType = 'Nothing'

calculator2 = Calculator(registrationName='Calculator2', Input=computeDerivatives1)
calculator2.AttributeType = 'Cell Data'
calculator2.Function = 'mag(Vorticity)'

cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=calculator2)
cellDatatoPointData1.CellDataArraytoprocess = ['Result']

contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
contour1.ContourBy = ['POINTS', 'Result']
contour1.Isosurfaces = [40]
contour1.PointMergeMethod = 'Uniform Binning'

resultLUT = GetColorTransferFunction('Result')

display = Show(contour1, view, 'GeometryRepresentation')
display.Representation = 'Surface'
display.ColorArrayName = ['POINTS', 'Result']
display.LookupTable = resultLUT
display.SelectTCoordArray = 'None'
display.SelectNormalArray = 'Normals'
display.SelectTangentArray = 'None'
display.OSPRayScaleArray = 'Result'
display.OSPRayScaleFunction = 'PiecewiseFunction'
display.SelectOrientationVectors = 'Vorticity'
display.ScaleFactor = 11.741285705566407
display.SelectScaleArray = 'Result'
display.GlyphType = 'Arrow'
display.GlyphTableIndexArray = 'Result'
display.GaussianRadius = 0.5870642852783203
display.SetScaleArray = ['POINTS', 'Result']
display.ScaleTransferFunction = 'PiecewiseFunction'
display.OpacityArray = ['POINTS', 'Result']
display.OpacityTransferFunction = 'PiecewiseFunction'
display.DataAxesGrid = 'GridAxesRepresentation'
display.PolarAxes = 'PolarAxesRepresentation'
display.SelectInputVectors = ['POINTS', 'Vorticity']
display.WriteLog = ''

display.ScaleTransferFunction.Points = [1.641363501548767, 0.0, 0.5, 0.0, 1.641607642173767, 1.0, 0.5, 0.0]
display.OpacityTransferFunction.Points = [1.641363501548767, 0.0, 0.5, 0.0, 1.641607642173767, 1.0, 0.5, 0.0]
display.SetScalarBarVisibility(view, False)
resultPWF = GetOpacityTransferFunction('Result')
resultTF2D = GetTransferFunction2D('Result')
ColorBy(display, None)
HideScalarBarIfNotNeeded(resultLUT, view)
display.AmbientColor = [0.3333333333333333, 0.0, 1.0]
display.DiffuseColor = [0.3333333333333333, 0.0, 1.0]
Hide(cellDatatoPointData1, view)
view.OrientationAxesVisibility = 0


view.Update()

view.CameraPosition = [445.8245175004483, 310.69353937581025, 416.6850538377703]
view.CameraFocalPoint = [63.500000000000014, 63.5, 63.50000000000003]
view.CameraViewUp = [-0.28038806220484247, 0.9021477885353584, -0.32789007641856993]
view.CameraParallelScale = 151.5559135019442


extractor = CreateExtractor('VTPD', producer, registrationName='VTPD')
options = catalyst.Options()
options.ExtractsOutputDirectory = './dst_r1_N256'
options.GlobalTrigger.Frequency = 1


def catalyst_execute(info):
    global producer

    contour1.UpdatePipeline()
    view.Update()

    SaveExtractsUsingCatalystOptions(options)

    fname = "dst_r1_N256/isosurface_curl-%d.png" % info.timestep
    ResetCamera()
    SaveScreenshot(fname, ImageResolution=(1024, 1024))
