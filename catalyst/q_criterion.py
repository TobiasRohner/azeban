from paraview.simple import *

from paraview.catalyst import get_args, get_execute_params

# print values for parameters passed via adaptor (note these don't change,
# and hence must be created as command line params)
print("executing catalyst_pipeline")
print("===================================")
print("pipeline args={}".format(get_args()))
print("===================================")

# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")


view = CreateRenderView()

gradient = Gradient(registrationName='Gradient', Input=producer)
gradient.ScalarArray = ['POINTS', 'velocity']
gradient.ComputeGradient = 0
gradient.ComputeVorticity = 1
gradient.ComputeQCriterion = 1

contour = Contour(registrationName='Contour', Input=gradient)
contour.ContourBy = ['POINTS', 'Q Criterion']
contour.PointMergeMethod = 'Uniform Binning'
contour.Isosurfaces = [0.005, -0.005]

display = Show(contour, view, 'GeometryRepresentation')

# init the 'Plane' selected for 'SliceFunction'
#display.SliceFunction.Origin = [31.5, 31.5, 31.5]
ColorBy(display, ('CELLS', 'Vorticity', 'Magnitude'))
display.SetRepresentationType('Volume')
display.RescaleTransferFunctionToDataRange(True, True)
display.SetScalarBarVisibility(view, True)
display.Representation = 'Surface'
display.ColorArrayName = [None, '']
display.SelectTCoordArray = 'None'
display.SelectNormalArray = 'None'
display.SelectTangentArray = 'None'
display.OSPRayScaleFunction = 'PiecewiseFunction'
display.SelectOrientationVectors = 'None'
display.ScaleFactor = -2.0000000000000002e+298
display.SelectScaleArray = 'None'
display.GlyphType = 'Arrow'
display.GlyphTableIndexArray = 'None'
display.GaussianRadius = -1e+297
display.SetScaleArray = [None, '']
display.ScaleTransferFunction = 'PiecewiseFunction'
display.OpacityArray = [None, '']
display.OpacityTransferFunction = 'PiecewiseFunction'
display.DataAxesGrid = 'GridAxesRepresentation'
display.PolarAxes = 'PolarAxesRepresentation'
display.SelectInputVectors = [None, '']
display.WriteLog = ''

view.Update()

view.CameraPosition = [-81.31900263997017, 137.0814101668189, 174.89444702020845]
view.CameraFocalPoint = [31.500000000000064, 31.50000000000005, 31.500000000000046]
view.CameraViewUp = [0.2733768161528506, 0.8646225876125255, -0.4215363535691214]
view.CameraParallelScale = 54.559600438419636


def catalyst_execute(info):
    global producer

    contour.UpdatePipeline()
    view.Update()
    #print(dict(producer))

    fname = "output-%d.png" % info.timestep
    ResetCamera()
    SaveScreenshot(fname, ImageResolution=(1920, 1080))
