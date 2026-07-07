mujoco_version "3.10.0"

struct Quat { w : double  x : double  y : double  z : double }

variant Orientation {
  quat : Quat
}

element Body2 (xml="body") {
  name : string
}

element Thing {
  name  : string (required)
  angle : double (unit=angle)
  tag   : string (xml="class")
  orient : variant Orientation (variant_group=orient, variant_tag=quat)
  text  : string (element_text)
}
