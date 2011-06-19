#include <vector>
#include <math.h>

#include <ode/ode.h>
#include <drawstuff/drawstuff.h>


#ifdef dDOUBLE
# define dsDrawCylinder dsDrawCylinderD
# define dsDrawSphere dsDrawSphereD
#endif


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

static const dReal lazer_length = 1.5;
static const dReal lazer_radius = 0.002;
static const dReal lazer_vel = 0.1;
static dBody* lazer_body;
static dGeom* lazer_geom;

static const dReal scanned_radius = 0.35;
static dBody* scanned_body;
static dGeom* scanned_geom;

// simulation

static void simule_lazer(void)
{
  // move the lazer up
  const dReal* const pos = lazer_body->getPosition();
  const dReal* const vel = lazer_body->getLinearVel();

  if ((pos[2] <= (2 * hax_radius + 0.02)) || (pos[2] >= (vax_length)))
    lazer_body->setLinearVel(vel[0], vel[1], vel[2] * -1);
}

static inline dReal rps_to_vel(dReal);
__attribute__((unused))
static void simule_rotor(void)
{
#if 0

  // wait for stabilized speed
  static unsigned int pass = 0;
  if (++pass < 20) return ;

  static dReal prev_vel = 0;

  // correct body position and rotation
  const dReal* const vel = rax_body->getAngularVel();

  if (pass == 20) prev_vel = vel[2];

  dReal new_vel = vel[2];
  if (fabs(new_vel - prev_vel) > 0.01)
  {
    printf("diff: %f\n", new_vel - prev_vel);
    new_vel += new_vel - prev_vel;
    rax_body->setAngularVel(vel[0], vel[1], new_vel);
  }

  prev_vel = new_vel;

#endif

//   dBodyID bodyID = rax_body->id();
//   const dReal *rot = dBodyGetAngularVel( bodyID );
//   const dReal *quat_ptr;
//   dReal quat[4], quat_len;
//   quat_ptr = dBodyGetQuaternion( bodyID );
//   quat[0] = quat_ptr[0];
//   quat[1] = 0;
//   quat[2] = 0; 
//   quat[3] = quat_ptr[3]; 
//   quat_len = sqrt( quat[0] * quat[0] + quat[3] * quat[3] );
//   quat[0] /= quat_len;
//   quat[3] /= quat_len;
//   dBodySetQuaternion( bodyID, quat );
//   dBodySetAngularVel( bodyID, 0, 0, rot[2] );
}


typedef dReal real_type;

typedef struct triple
{
  real_type height; // ray height
  real_type depth; // ray depth
  real_type alpha; // axis angle
} triple_t;

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
  // r the rotation matrix
  // atan2 returns in [-pi, pi]
  const real_type a = atan2(r[4], r[0]);
  return a + M_PI;
}

static inline dReal rtod(dReal);

static void simule_sampling(void)
{
  static const int contact_size = 32;

  dContactGeom contacts[contact_size];
  
  const int contact_count = dCollide
    (*lazer_geom, *scanned_geom, contact_size, contacts, sizeof(dContactGeom));

  if (contact_count == 0) return ;

  const dReal* const rax_rot = rax_body->getRotation();
  const dReal* const vax_pos = vax_body->getPosition();
  const dReal* const lazer_body_pos = lazer_body->getPosition();
  // the cyliner pos must be converted to the lazer pos
  real_type lazer_pos[3] = { vax_pos[0], vax_pos[1], lazer_body_pos[2] };

  printf("contact_count == %d\n", contact_count);
  printf("{\n");

  for (int i = 0; i < contact_count; ++i)
  {
    dContactGeom& c = contacts[i];

    // triple
    const real_type h = lazer_pos[2];
    const real_type a = get_z_angle(rax_rot);
    const real_type d = compute_distance(lazer_pos, c.pos);

    printf("%f %f %f\n", h, rtod(a), d);
  }

  printf("}\n");
}

static void simule(void)
{
  simule_rotor();
  simule_lazer();

  static unsigned int pass = 0;
  if ((++pass & (8 - 1)) == 0)
    simule_sampling();

  // stepsize in second
  dWorldStep(*world, 0.05);
}


// drawing

static void draw_scanned(void)
{
  dsSetColor(0.4, 0.4, 0);
  dsDrawSphere
    (scanned_body->getPosition(), scanned_body->getRotation(), scanned_radius);
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

static void draw_lazer(void)
{
  dsSetColor(1.0f, 0.0f, 0.0f);
  dsDrawCylinder
    (lazer_body->getPosition(), lazer_body->getRotation(), lazer_length, lazer_radius);
}

static void redraw(void)
{
  draw_hax();
  draw_vax();
  draw_rax();
  draw_sax();
  draw_scanned();
  draw_lazer();
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
    rax_body->setAngularVel(0, 0, rps_to_vel(5));
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

  // create lazer
  {
    lazer_body = new dBody(*world);
    dMass mass;
    mass.setCylinderTotal(0.000001, 3, lazer_radius, lazer_length);
    lazer_body->setMass(mass);

    // lazer_geom = new dCylinder(*space, lazer_radius, lazer_length);
    lazer_geom = new dRay(*space, lazer_length);
    lazer_geom->setBody(*lazer_body);

    dMatrix3 R;
    dRSetIdentity(R);
    dRFromAxisAndAngle(R, 1, 0, 0, dtor(90));
    lazer_body->setRotation(R);
    lazer_body->setPosition(0, -0.25, vax_length / 2);
    lazer_body->setAngularVel(0, 0, 0);
    lazer_body->setLinearVel(0, 0, lazer_vel);
  }

  // create the scanned object
  {
    scanned_body = new dBody(*world);
    dMass mass;
    mass.setSphereTotal(0.00001, scanned_radius);
    scanned_body->setMass(mass);

    scanned_geom = new dSphere(*space, scanned_radius);
    scanned_geom->setBody(*scanned_body);

    dMatrix3 R;
    dRSetIdentity(R);
    scanned_body->setRotation(R);
    scanned_body->setPosition(0, 0, vax_length / 2);
    scanned_body->setAngularVel(0, 0, 0);
    scanned_body->setLinearVel(0, 0, 0);
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
  // create vax, lazer slide joint
  {
    dJointID joint = dJointCreateSlider(*world, 0);
    dJointAttach(joint, *vax_body, *lazer_body);
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
