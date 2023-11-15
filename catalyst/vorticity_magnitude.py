# state file generated using paraview version 5.11.0
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1154, 813]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [0.49609375, 0.49609375, 0.49609375]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-0.7870890564546027, -1.739766242972274, -0.44330775116343846]
renderView1.CameraFocalPoint = [0.4960937500000009, 0.4960937499999993, 0.49609374999999956]
renderView1.CameraViewUp = [-0.8831871086598764, 0.4463681662279397, 0.14399996970594992]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.8592595803173727

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1154, 813)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# registrationName must match the channel name used in the
# 'CatalystAdaptor'.
producer = TrivialProducer(registrationName="grid")

# create a new 'Gradient'
gradient1 = Gradient(registrationName='Gradient1', Input=producer)
gradient1.ScalarArray = ['POINTS', 'velocity']
gradient1.ComputeGradient = 0
gradient1.ComputeVorticity = 1

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from gradient1
gradient1Display = Show(gradient1, renderView1, 'UniformGridRepresentation')

# get 2D transfer function for 'Vorticity'
vorticityTF2D = GetTransferFunction2D('Vorticity')
vorticityTF2D.ScalarRangeInitialized = 1
vorticityTF2D.Range = [0.00035161785811201876, 384.0, 0.0, 1.0]

# get color transfer function/color map for 'Vorticity'
vorticityLUT = GetColorTransferFunction('Vorticity')
vorticityLUT.AutomaticRescaleRangeMode = 'Never'
vorticityLUT.TransferFunction2D = vorticityTF2D
vorticityLUT.RGBPoints = [0.00035161785811201876, 0.231373, 0.298039, 0.752941, 192.00017580892907, 0.865003, 0.865003, 0.865003, 384.0, 0.705882, 0.0156863, 0.14902]
vorticityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Vorticity'
vorticityPWF = GetOpacityTransferFunction('Vorticity')
vorticityPWF.Points = [0.00035161785811201876, 0.0, 0.5, 0.0, 384.0, 1, 0.5, 0.0]
vorticityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
gradient1Display.Representation = 'Volume'
gradient1Display.ColorArrayName = ['POINTS', 'Vorticity']
gradient1Display.LookupTable = vorticityLUT
gradient1Display.SelectTCoordArray = 'None'
gradient1Display.SelectNormalArray = 'None'
gradient1Display.SelectTangentArray = 'None'
gradient1Display.OSPRayScaleArray = 'Vorticity'
gradient1Display.OSPRayScaleFunction = 'PiecewiseFunction'
gradient1Display.SelectOrientationVectors = 'velocity'
gradient1Display.ScaleFactor = 0.09921875000000001
gradient1Display.SelectScaleArray = 'None'
gradient1Display.GlyphType = 'Arrow'
gradient1Display.GlyphTableIndexArray = 'None'
gradient1Display.GaussianRadius = 0.0049609375
gradient1Display.SetScaleArray = ['POINTS', 'Vorticity']
gradient1Display.ScaleTransferFunction = 'PiecewiseFunction'
gradient1Display.OpacityArray = ['POINTS', 'Vorticity']
gradient1Display.OpacityTransferFunction = 'PiecewiseFunction'
gradient1Display.DataAxesGrid = 'GridAxesRepresentation'
gradient1Display.PolarAxes = 'PolarAxesRepresentation'
gradient1Display.ScalarOpacityFunction = vorticityPWF
gradient1Display.ScalarOpacityUnitDistance = 0.013531646934131851
gradient1Display.SelectInputVectors = ['POINTS', 'velocity']
gradient1Display.WriteLog = ''
#gradient1Display.InputVectors = ['POINTS', 'velocity']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gradient1Display.ScaleTransferFunction.Points = [-665.1116943359375, 0.0, 0.5, 0.0, 370.2078857421875, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gradient1Display.OpacityTransferFunction.Points = [-665.1116943359375, 0.0, 0.5, 0.0, 370.2078857421875, 1.0, 0.5, 0.0]

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup extractors
# ----------------------------------------------------------------

# create extractor
pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
# trace defaults for the extractor.
pNG1.Trigger = 'TimeStep'

# init the 'PNG' selected for 'Writer'
pNG1.Writer.FileName = 'anim_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [2048, 2048]
pNG1.Writer.Format = 'PNG'

# ----------------------------------------------------------------
# restore active source
SetActiveSource(pNG1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')


def catalyst_execute(info):
    global producer

    gradient1.UpdatePipeline()
    renderView1.Update()

    #ResetCamera()
    fname = f'dst_r0_N512/anim.png'
    SaveScreenshot(fname, ImageResolution=(1024, 1024))
