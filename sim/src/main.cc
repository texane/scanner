// static configuration
#define CONFIG_SCANNED_SPHERE 0
#define CONFIG_SCANNED_BOX 0
#define CONFIG_SCANNED_BUNNY 1
#define CONFIG_SCANNED_MESH CONFIG_SCANNED_BUNNY
#define CONFIG_DRAW_CONTACTS 1
#define CONFIG_USE_RAY 1

#include <vector>
#include <math.h>

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>

#if CONFIG_SCANNED_BUNNY
# include "bunny_geom.h"
#endif


#ifdef dDOUBLE
# define dsDrawCylinder dsDrawCylinderD
# define dsDrawSphere dsDrawSphereD
# define dsDrawTriangle dsDrawTriangleD
#endif


typedef dReal real_type;


// sharp ir sensor

// device model
// sampling latency (real_type usecs)
__attribute__((unused))
static real_type sharpir_dtov(real_type d)
{
  // return the voltage given the distance
  return 0;
}


static void start(void)
{
  static float xyz[3] = {-2, 0, 1.50};
  static float hpr[3] = {0.0f, -30, 0.0f};
  dsSetViewpoint(xyz, hpr);
}


// command callback

static void command(int)
{
}


// world globals

static dWorld* world;
static dSpace* space;
static dPlane* flor;
static dsFunctions funcs;

static const dReal vax_radius = 0.01;
static const dReal vax_length = 1;
static dBody* vax_body;

static const dReal hax_radius = 0.01;
static const dReal hax_length = 1;
static dBody* hax_body;

static const dReal rax_radius = 0.01;
static const dReal rax_length = 0.1;
static dBody* rax_body;
static dGeom* rax_geom;

static const dReal sax_radius = 0.03;
static const dReal sax_length = 0.01;
static dBody* sax_body;

static const dReal ray_length = 1.5;
static const dReal ray_radius = 0.001;
static const dReal ray_vel = 0.3;
static dBody* ray_body;
static dGeom* ray_geom;

#if (CONFIG_SCANNED_SPHERE || CONFIG_SCANNED_BOX)
static const dReal scanned_radius = 0.2;
static dBody* scanned_body;
static dGeom* scanned_geom;
#else
static dTriMeshDataID bunny_data;
static dGeomID scanned_geom;
#endif


#if CONFIG_DRAW_CONTACTS

#include <list>

std::list<dBody*> contact_bodies;

static const real_type contact_radius = 0.05;

static void add_contact(const real_type* pos)
{
  dBody* const body = new dBody(*world);
  dMass mass;
  mass.setBoxTotal
    (0.00001, contact_radius, contact_radius, contact_radius);
  body->setMass(mass);

  dGeom* const geom = new dBox
    (*space, contact_radius, contact_radius, contact_radius);
  geom->setBody(*body);

  dMatrix3 R;
  dRSetIdentity(R);
  body->setRotation(R);
  body->setPosition(pos[0], pos[1], pos[2]);
  body->setAngularVel(0, 0, 0);
  body->setLinearVel(0, 0, 0);

  contact_bodies.push_back(body);
}

static void draw_contacts(void)
{
  dVector3 sides = { contact_radius, contact_radius, contact_radius };

  dsSetColor(0, 1, 0);

  std::list<dBody*>::const_iterator pos = contact_bodies.begin();
  std::list<dBody*>::const_iterator end = contact_bodies.end();
  for (; pos != end; ++pos)
    dsDrawBox ((*pos)->getPosition(), (*pos)->getRotation(), sides);
}

#endif // CONFIG_DRAW_CONTACTS


// simulation

static void simule_ray(void)
{
  // move the ray up
  const dReal* const pos = ray_body->getPosition();
  const dReal* const vel = ray_body->getLinearVel();

  if ((pos[2] <= (2 * hax_radius + 0.02)) || (pos[2] >= (vax_length)))
    ray_body->setLinearVel(vel[0], vel[1], vel[2] * -1);
}

static inline void ahdr_to_xyz
(const real_type* ahdr, real_type* xyz)
{
  // hadr the angle, heigth, depth, radius tuple
  // xyz the absolute cartesian coords

  const real_type d = ahdr[3] - ahdr[2];

  xyz[0] = d * cos(ahdr[0]);
  xyz[1] = ahdr[1];
  xyz[2] = d * sin(ahdr[0]);
}

static inline real_type compute_distance
(const real_type* a, const real_type* b)
{
  const real_type dx = a[0] - b[0];
  const real_type dy = a[1] - b[1];
  const real_type dz = a[2] - b[2];
  return sqrt(dx * dx + dy * dy + dz * dz);
}

static inline real_type get_z_angle(const real_type* r)
{
  // z angle from rotation matrix r
  // http://www.codeguru.com/forum/archive/index.php/t-329530.html

  // atan2 returns in [-pi, pi]
  return atan2(r[4], r[0]) + M_PI;
}

static inline dReal rtod(dReal);

static void simule_sampling(void)
{
  static const int contact_size = 32;

  dContactGeom contacts[contact_size];
  
  const int contact_count = dCollide
#if (CONFIG_SCANNED_BOX || CONFIG_SCANNED_SPHERE)
    (*ray_geom, *scanned_geom, contact_size, contacts, sizeof(dContactGeom));
#elif CONFIG_SCANNED_MESH
    (*ray_geom, scanned_geom, contact_size, contacts, sizeof(dContactGeom));
#endif

  if (contact_count == 0) return ;

  const dReal* const rax_rot = rax_body->getRotation();
  const dReal* const vax_pos = vax_body->getPosition();
  const dReal* const ray_body_pos = ray_body->getPosition();
  // the cyliner pos must be converted to the ray pos
  real_type ray_pos[3] = { vax_pos[0], vax_pos[1], ray_body_pos[2] };

  // filter and find the nearest one
  real_type nearest_d = 0;
  int nearest_i = -1;
  for (int i = 0; i < contact_count; ++i)
  {
    dContactGeom& c = contacts[i];

#if 0 // unused
    const real_type* const body_pos = dGeomGetPosition(c.g1);
    real_type contact_pos[3];
    contact_pos[0] = body_pos[0] + c.normal[0] * c.depth;
    contact_pos[1] = body_pos[1] + c.normal[1] * c.depth;
    contact_pos[2] = body_pos[2] + c.normal[2] * c.depth;
#endif // unused

    const real_type d = compute_distance(ray_pos, c.pos);
    if ((nearest_i == -1) || (d < nearest_d))
    {
      nearest_i = i;
      nearest_d = d;
    }
  }

  // not found
  if (nearest_i == -1) return ;

  // ahdr tuple
  real_type ahdr[4];
  ahdr[0] = get_z_angle(rax_rot);
  ahdr[1] = ray_pos[2];
  ahdr[2] = nearest_d;
  ahdr[3] = hax_length;

  // convert to xyz
  real_type xyz[3];
  ahdr_to_xyz(ahdr, xyz);

  printf("%f %f %f\n", xyz[0], xyz[1], xyz[2]);
  fflush(stdout);

#if CONFIG_DRAW_CONTACTS
  // turn xyz into the world coords
  real_type xyz_trans[3] = { -xyz[2], xyz[0], xyz[1] };
  add_contact(xyz_trans);
#endif // CONFIG_DRAW_CONTACTS
}

static void simule(void)
{
  simule_ray();
  simule_sampling();

  // stepsize in second
  dWorldStep(*world, 0.05);
}


// drawing

#if CONFIG_SCANNED_MESH

static void scale_mesh(real_type ratio)
{
  float* v = Vertices;
  for (int count = VertexCount * 3; count; --count, ++v) *v *= ratio;
}

static void draw_mesh(dGeomID geom)
{
  dTriIndex* Indices = (dTriIndex*)TriIndices;

  // assume all trimeshes are drawn as bunnies
  const dReal* const Pos = dGeomGetPosition(geom);
  const dReal* const Rot = dGeomGetRotation(geom);

  for (int ii = 0; ii < IndexCount / 3; ii++)
  {
    const dReal v[9] =
    {
      // explicit conversion from float to dReal
      Vertices[Indices[ii * 3 + 0] * 3 + 0],
      Vertices[Indices[ii * 3 + 0] * 3 + 1],
      Vertices[Indices[ii * 3 + 0] * 3 + 2],
      Vertices[Indices[ii * 3 + 1] * 3 + 0],
      Vertices[Indices[ii * 3 + 1] * 3 + 1],
      Vertices[Indices[ii * 3 + 1] * 3 + 2],
      Vertices[Indices[ii * 3 + 2] * 3 + 0],
      Vertices[Indices[ii * 3 + 2] * 3 + 1],
      Vertices[Indices[ii * 3 + 2] * 3 + 2]
    };

    dsDrawTriangle(Pos, Rot, &v[0], &v[3], &v[6], 1);
  }
}

#endif // CONFIG_SCANNED_MESH

static void draw_scanned(void)
{
  dsSetColor(0.9, 0.9, 0.9);
#if CONFIG_SCANNED_SPHERE
  dsDrawSphere
    (scanned_body->getPosition(), scanned_body->getRotation(), scanned_radius);
#elif CONFIG_SCANNED_BOX
  dVector3 sides;
  dGeomBoxGetLengths(*scanned_geom, sides);
  dsDrawBox  
    (scanned_body->getPosition(), scanned_body->getRotation(), sides);
#elif CONFIG_SCANNED_MESH
  draw_mesh(scanned_geom);
#endif
}

static void draw_vax(void)
{
  dsSetColor(0, 0.2, 0.8f);
  dsDrawCylinder
    (vax_body->getPosition(), vax_body->getRotation(), vax_length, vax_radius);
}

static void draw_hax(void)
{
  dsSetColor(0, 0.2, 0.8f);
  dsDrawCylinder
    (hax_body->getPosition(), hax_body->getRotation(), hax_length, hax_radius);
}

static void draw_rax(void)
{
  dsSetColor(0.2, 0.2, 0.2);
  dsDrawCylinder
    (rax_body->getPosition(), rax_body->getRotation(), rax_length, rax_radius);
}

static void draw_sax(void)
{
  dsSetColor(0.2, 0.2, 0.2);
  dsDrawCylinder
    (sax_body->getPosition(), sax_body->getRotation(), sax_length, sax_radius);
}

static void draw_ray(void)
{
  dsSetColor(1.0f, 0.0f, 0.0f);

#if CONFIG_USE_RAY
  dReal len;
  dVector3 pos;
  dVector3 dir;
  len = dGeomRayGetLength(ray_geom->id());
  dGeomRayGet(ray_geom->id(), pos, dir);
  pos[0] += dir[0] * len / 2;
  pos[1] += dir[1] * len / 2;
  pos[2] += dir[2] * len / 2;
  dsDrawCylinder
    (pos, ray_body->getRotation(), ray_length, ray_radius);
#else
  dsDrawCylinder
    (ray_body->getPosition(), ray_body->getRotation(), ray_length, ray_radius);
#endif
}

static void redraw(void)
{
  draw_hax();
  draw_vax();
  draw_rax();
  draw_sax();
  draw_scanned();
  draw_ray();

#if CONFIG_DRAW_CONTACTS
  draw_contacts();
#endif // CONFIG_DRAW_CONTACTS
}

static void step(int)
{
  simule();
  redraw();
}

__attribute__((unused))
static inline dReal dtor(dReal d)
{
  return (M_PI * d) / 180.f;
}

__attribute__((unused))
static inline dReal rtod(dReal r)
{
  return (180.f * r) / M_PI;
}

static inline dReal rps_to_vel(dReal rps)
{
  // round per second to angular velocity
  // return the angular velocity, in degree per second
  // rps the round per second
  
  return rps * 360;
}

static void initialize(void)
{
  dInitODE2(0);

  funcs.version = DS_VERSION;
  funcs.start = start;
  funcs.step = step;
  funcs.command = command;
  funcs.stop = 0;
  funcs.path_to_textures = "../dat/drawstuff/textures";

  world = new dWorld();
  world->setGravity(0, 0, 0);
  world->setCFM(1e-5f);
  world->setLinearDamping(0.00001f);
  world->setAngularDamping(0.0001f);

  space = new dSimpleSpace(0);

  flor = new dPlane(*space, 0, 0, 1, 0);

  // create vertical axe
  {
    dMass mass;
    mass.setCylinderTotal(0.000001, 3, vax_radius, vax_length);
    vax_body = new dBody(*world);
    vax_body->setMass(mass);

    dGeom* const geom = new dCylinder(*space, vax_radius, vax_length);
    geom->setBody(*vax_body);

    dMatrix3 R;
    dRSetIdentity(R);
    vax_body->setRotation(R);
    vax_body->setPosition(0, -1, vax_length / 2 + hax_radius + 0.02);
    vax_body->setAngularVel(0, 0, 0);
    vax_body->setLinearVel(0, 0, 0);
  }

  // create horizontal axe
  {
    hax_body = new dBody(*world);
    dMass mass;
    mass.setCylinderTotal(0.000001, 3, hax_radius, hax_length);
    hax_body->setMass(mass);

    dGeom* const geom = new dCylinder(*space, hax_radius, hax_length * 2);
    geom->setBody(*hax_body);

    dMatrix3 R;
    dRSetIdentity(R);
    dRFromAxisAndAngle(R, 1, 0, 0, dtor(90));
    hax_body->setRotation(R);
    hax_body->setPosition(0, - hax_length / 2, hax_radius + 0.02);
    hax_body->setAngularVel(0, 0, 0);
    hax_body->setLinearVel(0, 0, 0);
  }

  // create the rotor axis
  {
    rax_body = new dBody(*world);
    dMass mass;
    mass.setCylinderTotal(1, 3, rax_radius, rax_length);
    rax_body->setMass(mass);

    rax_geom = new dCylinder(*space, rax_radius, rax_length);
    rax_geom->setBody(*rax_body);

    dMatrix3 R;
    dRSetIdentity(R);
    rax_body->setRotation(R);
    rax_body->setPosition(0, 0, rax_length / 2);
    rax_body->setAngularVel(0, 0, rps_to_vel(1));
    rax_body->setLinearVel(0, 0, 0);
  }

  // create the stator
  {
    dMass mass;
    mass.setCylinderTotal(300, 3, sax_radius, sax_length);
    sax_body = new dBody(*world);
    sax_body->setMass(mass);

    dGeom* const geom = new dCylinder(*space, sax_radius, sax_length);
    geom->setBody(*sax_body);

    dMatrix3 R;
    dRSetIdentity(R);
    sax_body->setRotation(R);
    hax_body->setPosition(0, - hax_length / 2, hax_radius + 0.02);
    sax_body->setAngularVel(0, 0, 0);
    sax_body->setLinearVel(0, 0, 0);
  }

  // create ray
  {
    ray_body = new dBody(*world);
    dMass mass;
    mass.setCylinderTotal(0.000001, 3, ray_radius, ray_length);
    ray_body->setMass(mass);

#if CONFIG_USE_RAY
    ray_geom = new dRay(*space, ray_length);
#else
    ray_geom = new dCylinder(*space, ray_radius, ray_length);
#endif // CONFIG_USE_RAY
    ray_geom->setBody(*ray_body);

    dMatrix3 R;
    dRSetIdentity(R);
    dRFrom2Axes(R, 1, 0, 0, 0, 0, -1);
    ray_body->setRotation(R);
#if CONFIG_USE_RAY
    ray_body->setPosition(0, -hax_length, vax_length / 2);
#else
    ray_body->setPosition(0, -0.25, vax_length / 2);
#endif
    ray_body->setAngularVel(0, 0, 0);
    ray_body->setLinearVel(0, 0, ray_vel);

#if CONFIG_USE_RAY
    // adjust the ray
    dVector3 pos, dir;
    dGeomRayGet(*ray_geom, pos, dir);
    dir[1] = -1;
    dGeomRaySet(*ray_geom, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
    dGeomRayGet(*ray_geom, pos, dir);
    printf("%f %f %f, %f %f %f\n", pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);
#endif // CONFIG_USE_RAY
  }

  // create the scanned object
  {
#if (CONFIG_SCANNED_SPHERE || CONFIG_SCANNED_BOX)
    scanned_body = new dBody(*world);
    dMass mass;

# if CONFIG_SCANNED_SPHERE
    mass.setSphereTotal(0.00001, scanned_radius);
    scanned_body->setMass(mass);
    scanned_geom = new dSphere(*space, scanned_radius);
# elif CONFIG_SCANNED_BOX
    mass.setBoxTotal
      (0.00001, scanned_radius, scanned_radius, scanned_radius);
    scanned_body->setMass(mass);
    scanned_geom = new dBox
      (*space, scanned_radius, scanned_radius, scanned_radius);
#endif // CONFIG_SCANNED_BOX

    scanned_geom->setBody(*scanned_body);
    dMatrix3 R;
    dRSetIdentity(R);
    scanned_body->setRotation(R);
    scanned_body->setPosition(0, 0, vax_length / 2);
    scanned_body->setAngularVel(0, 0, 0);
    scanned_body->setLinearVel(0, 0, 0);

#elif CONFIG_SCANNED_MESH
    scale_mesh(0.5);

    bunny_data = dGeomTriMeshDataCreate();
    dGeomTriMeshDataBuildSingle
    (
     bunny_data,
     &Vertices[0],
     3 * sizeof(float),
     VertexCount,
     (dTriIndex*)TriIndices,
     IndexCount, 3 * sizeof(dTriIndex)
    );
    scanned_geom = dCreateTriMesh(*space, bunny_data, 0, 0, 0);
    dGeomSetData(scanned_geom, bunny_data);

    dMatrix3 R;
    dRSetIdentity(R);
    dRFromAxisAndAngle(R, 1, 0, 0, dtor(90));
    dGeomSetRotation(scanned_geom, R);
    dGeomSetPosition(scanned_geom, 0, 0, vax_length / 2);
#endif

  }

#if 0  // create sax, env fixed joint
  {
    dJointID sax_env_joint = dJointCreateFixed(*world, 0);
    dJointAttach(sax_env_joint, *rax_body, 0);
    dJointSetFixed(sax_env_joint);
  }
#endif

#if 1
  // create sax, rax fixed joint
  {
    dJointID rax_sax_joint = dJointCreateFixed(*world, 0);
    dJointAttach(rax_sax_joint, *sax_body, *rax_body);
    dJointSetFixed(rax_sax_joint);
  }
#endif

#if 0 // rax plan2d joint
  {
    dJointID planeJointID = dJointCreatePlane2D( *world, 0);
    dJointAttach( planeJointID, *hax_body, 0 );
    dJointSetPlane2DXParam (planeJointID, dParamFMax, 0);
    dJointSetPlane2DYParam (planeJointID, dParamFMax, 0);
  }
#endif

#if 1
  // create rax, hax fixed joint
  {
    dJointID rax_hax_joint = dJointCreateFixed(*world, 0);
    dJointAttach(rax_hax_joint, *rax_body, *hax_body);
    dJointSetFixed(rax_hax_joint);
  }
#endif

#if 0
  // create rax, hax hinge joint
  {
    dJointID rax_hax_joint = dJointCreateHinge(*world, 0);
    dJointAttach(rax_hax_joint, *rax_body, *hax_body);
    dJointSetHingeAnchor(rax_hax_joint, 0, 0, 0);
    dJointSetHingeAxis(rax_hax_joint, 0, 0, 1);
    dJointSetHingeParam(rax_hax_joint, dParamFMax, 1);
  }
#endif

#if 1
  // create hax, vax fixed joint
  {
    dJointID vax_hax_joint = dJointCreateFixed(*world, 0);
    dJointAttach(vax_hax_joint, *vax_body, *hax_body);
    dJointSetFixed(vax_hax_joint);
  }
#endif

#if 1
  // create hax, vax fixed joint
  {
    dJointID vax_hax_joint = dJointCreateFixed(*world, 0);
    dJointAttach(vax_hax_joint, *vax_body, *hax_body);
    dJointSetFixed(vax_hax_joint);
  }
#endif

#if 1
  // create vax, ray slide joint
  {
    dJointID joint = dJointCreateSlider(*world, 0);
    dJointAttach(joint, *vax_body, *ray_body);
    dJointSetSliderAxis(joint, 0, 0, 1);
  }
#endif
}

static void finalize(void)
{
  delete flor;
  delete world;
  delete space;

  dCloseODE();
}


// main

int main(int ac, char** av)
{
  initialize();
  dsSimulationLoop(ac, av, 512, 384, &funcs);
  finalize();

  return 0;
}
