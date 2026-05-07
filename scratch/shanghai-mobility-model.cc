/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Mobility model for Shanghai dataset trajectories
 */

#include "shanghai-mobility-model.h"
#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/integer.h"
#include "ns3/double.h"
#include "ns3/string.h"
#include "ns3/pointer.h"
#include <fstream>
#include <sstream>
#include <tinyxml2.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ShanghaiMobilityModel");

NS_OBJECT_ENSURE_REGISTERED (ShanghaiMobilityModel);

TypeId
ShanghaiMobilityModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ShanghaiMobilityModel")
    .SetParent<MobilityModel> ()
    .SetGroupName ("Mobility")
    .AddConstructor<ShanghaiMobilityModel> ()
    .AddAttribute ("TimeScale", "Time scale factor for trajectory playback",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&ShanghaiMobilityModel::SetSimulationTimeScale,
                                       &ShanghaiMobilityModel::GetSimulationTimeScale),
                   MakeDoubleChecker<double> ())
    ;
  return tid;
}

ShanghaiMobilityModel::ShanghaiMobilityModel ()
  : m_timeScale (1.0),
    m_startTime (0.0),
    m_trajectoryLoaded (false)
{
  NS_LOG_FUNCTION (this);
}

ShanghaiMobilityModel::~ShanghaiMobilityModel ()
{
  NS_LOG_FUNCTION (this);
}

void
ShanghaiMobilityModel::SetSimulationTimeScale (double scale)
{
  NS_LOG_FUNCTION (this << scale);
  m_timeScale = scale;
}

double
ShanghaiMobilityModel::GetSimulationTimeScale () const
{
  return m_timeScale;
}

void
ShanghaiMobilityModel::LoadTrajectoryFromOpenScenario (const std::string& filename)
{
  NS_LOG_FUNCTION (this << filename);
  ParseOpenScenarioFile (filename);
  m_trajectoryLoaded = true;
}

void
ShanghaiMobilityModel::LoadTrajectoryFromCSV (const std::string& filename)
{
  NS_LOG_FUNCTION (this << filename);
  ParseCSVFile (filename);
  m_trajectoryLoaded = true;
}

void
ShanghaiMobilityModel::SetVehicleTrajectory (const VehicleTrajectory& trajectory)
{
  NS_LOG_FUNCTION (this);
  m_trajectory = trajectory;
  m_trajectoryLoaded = true;
}

Vector
ShanghaiMobilityModel::DoGetPosition (void) const
{
  if (!m_trajectoryLoaded)
    {
      return Vector (0.0, 0.0, 0.0);
    }
  
  double currentTime = (Simulator::Now ().GetSeconds () - m_startTime) * m_timeScale;
  return InterpolatePosition (currentTime);
}

void
ShanghaiMobilityModel::DoSetPosition (const Vector &position)
{
  NS_LOG_FUNCTION (this << position);
  m_startTime = Simulator::Now ().GetSeconds ();
  NotifyCourseChange ();
}

Vector
ShanghaiMobilityModel::DoGetVelocity (void) const
{
  if (!m_trajectoryLoaded)
    {
      return Vector (0.0, 0.0, 0.0);
    }
  
  double currentTime = (Simulator::Now ().GetSeconds () - m_startTime) * m_timeScale;
  return InterpolateVelocity (currentTime);
}

void
ShanghaiMobilityModel::DoInitialize (void)
{
  NS_LOG_FUNCTION (this);
  MobilityModel::DoInitialize ();
  
  if (m_trajectoryLoaded)
    {
      m_startTime = Simulator::Now ().GetSeconds ();
      UpdatePosition ();
    }
}

void
ShanghaiMobilityModel::UpdatePosition (void)
{
  if (!m_trajectoryLoaded)
    {
      return;
    }
  
  NotifyCourseChange ();
  
  double currentTime = (Simulator::Now ().GetSeconds () - m_startTime) * m_timeScale;
  
  if (currentTime < m_trajectory.waypoints.back ().time)
    {
      m_updateEvent = Simulator::Schedule (Seconds (0.1), &ShanghaiMobilityModel::UpdatePosition, this);
    }
}

Vector
ShanghaiMobilityModel::InterpolatePosition (double time) const
{
  if (m_trajectory.waypoints.empty ())
    {
      return Vector (0.0, 0.0, 0.0);
    }
  
  if (time <= m_trajectory.waypoints[0].time)
    {
      return m_trajectory.waypoints[0].position;
    }
  
  if (time >= m_trajectory.waypoints.back ().time)
    {
      return m_trajectory.waypoints.back ().position;
    }
  
  for (size_t i = 0; i < m_trajectory.waypoints.size () - 1; ++i)
    {
      if (time >= m_trajectory.waypoints[i].time && time <= m_trajectory.waypoints[i + 1].time)
        {
          double t1 = m_trajectory.waypoints[i].time;
          double t2 = m_trajectory.waypoints[i + 1].time;
          Vector p1 = m_trajectory.waypoints[i].position;
          Vector p2 = m_trajectory.waypoints[i + 1].position;
          
          double alpha = (time - t1) / (t2 - t1);
          return Vector (p1.x + alpha * (p2.x - p1.x),
                         p1.y + alpha * (p2.y - p1.y),
                         p1.z + alpha * (p2.z - p1.z));
        }
    }
  
  return m_trajectory.waypoints.back ().position;
}

Vector
ShanghaiMobilityModel::InterpolateVelocity (double time) const
{
  if (m_trajectory.waypoints.size () < 2)
    {
      return Vector (0.0, 0.0, 0.0);
    }
  
  double dt = 0.1; // Time step for velocity calculation
  Vector pos1 = InterpolatePosition (time - dt);
  Vector pos2 = InterpolatePosition (time + dt);
  
  return Vector ((pos2.x - pos1.x) / (2 * dt),
                 (pos2.y - pos1.y) / (2 * dt),
                 (pos2.z - pos1.z) / (2 * dt));
}

void
ShanghaiMobilityModel::ParseOpenScenarioFile (const std::string& filename)
{
  NS_LOG_FUNCTION (this << filename);
  
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile (filename.c_str ()) != tinyxml2::XML_SUCCESS)
    {
      NS_LOG_ERROR ("Failed to load OpenSCENARIO file: " << filename);
      return;
    }
  
  // Clear existing trajectory
  m_trajectory.waypoints.clear ();
  
  // Find trajectory elements
  tinyxml2::XMLElement* root = doc.FirstChildElement ("OpenSCENARIO");
  if (!root)
    {
      NS_LOG_ERROR ("No OpenSCENARIO root element found");
      return;
    }
  
  // Parse vehicles and their trajectories
  tinyxml2::XMLElement* entities = root->FirstChildElement ("Entities");
  if (entities)
    {
      tinyxml2::XMLElement* scenarioObject = entities->FirstChildElement ("ScenarioObject");
      while (scenarioObject)
        {
          const char* name = scenarioObject->Attribute ("name");
          if (name)
            {
              m_trajectory.vehicleId = std::string (name);
            }
          
          tinyxml2::XMLElement* vehicle = scenarioObject->FirstChildElement ("Vehicle");
          if (vehicle)
            {
              const char* vehicleType = vehicle->Attribute ("vehicleCategory");
              if (vehicleType)
                {
                  m_trajectory.vehicleType = std::string (vehicleType);
                }
              
              tinyxml2::XMLElement* boundingBox = vehicle->FirstChildElement ("BoundingBox");
              if (boundingBox)
                {
                  tinyxml2::XMLElement* dimensions = boundingBox->FirstChildElement ("Dimensions");
                  if (dimensions)
                    {
                      dimensions->QueryDoubleAttribute ("length", &m_trajectory.length);
                      dimensions->QueryDoubleAttribute ("width", &m_trajectory.width);
                    }
                }
            }
          
          scenarioObject = scenarioObject->NextSiblingElement ("ScenarioObject");
        }
    }
  
  // Parse trajectory waypoints
  tinyxml2::XMLElement* story = root->FirstChildElement ("Story");
  while (story)
    {
      tinyxml2::XMLElement* act = story->FirstChildElement ("Act");
      while (act)
        {
          tinyxml2::XMLElement* maneuverGroup = act->FirstChildElement ("ManeuverGroup");
          while (maneuverGroup)
            {
              tinyxml2::XMLElement* maneuver = maneuverGroup->FirstChildElement ("Maneuver");
              while (maneuver)
                {
                  tinyxml2::XMLElement* event = maneuver->FirstChildElement ("Event");
                  while (event)
                    {
                      tinyxml2::XMLElement* action = event->FirstChildElement ("Action");
                      while (action)
                        {
                          tinyxml2::XMLElement* privateAction = action->FirstChildElement ("PrivateAction");
                          if (privateAction)
                            {
                              tinyxml2::XMLElement* routingAction = privateAction->FirstChildElement ("RoutingAction");
                              if (routingAction)
                                {
                                  tinyxml2::XMLElement* followTrajectoryAction = routingAction->FirstChildElement ("FollowTrajectoryAction");
                                  if (followTrajectoryAction)
                                    {
                                      tinyxml2::XMLElement* trajectory = followTrajectoryAction->FirstChildElement ("Trajectory");
                                      if (trajectory)
                                        {
                                          tinyxml2::XMLElement* shape = trajectory->FirstChildElement ("Shape");
                                          if (shape)
                                            {
                                              tinyxml2::XMLElement* polyline = shape->FirstChildElement ("Polyline");
                                              if (polyline)
                                                {
                                                  tinyxml2::XMLElement* vertex = polyline->FirstChildElement ("Vertex");
                                                  int waypointIndex = 0;
                                                  while (vertex)
                                                    {
                                                      TrajectoryPoint point;
                                                      tinyxml2::XMLElement* position = vertex->FirstChildElement ("Position");
                                                      if (position)
                                                        {
                                                          tinyxml2::XMLElement* worldPos = position->FirstChildElement ("WorldPosition");
                                                          if (worldPos)
                                                            {
                                                              worldPos->QueryDoubleAttribute ("x", &point.position.x);
                                                              worldPos->QueryDoubleAttribute ("y", &point.position.y);
                                                              worldPos->QueryDoubleAttribute ("z", &point.position.z);
                                                              
                                                              point.time = waypointIndex * 0.04; // 25 FPS = 0.04s per frame
                                                              m_trajectory.waypoints.push_back (point);
                                                            }
                                                        }
                                                      vertex = vertex->NextSiblingElement ("Vertex");
                                                      waypointIndex++;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                          action = action->NextSiblingElement ("Action");
                        }
                      event = event->NextSiblingElement ("Event");
                    }
                  maneuver = maneuver->NextSiblingElement ("Maneuver");
                }
              maneuverGroup = maneuverGroup->NextSiblingElement ("ManeuverGroup");
            }
          act = act->NextSiblingElement ("Act");
        }
      story = story->NextSiblingElement ("Story");
    }
  
  NS_LOG_INFO ("Loaded " << m_trajectory.waypoints.size () << " waypoints from " << filename);
}

void
ShanghaiMobilityModel::ParseCSVFile (const std::string& filename)
{
  NS_LOG_FUNCTION (this << filename);
  
  std::ifstream file (filename.c_str ());
  if (!file.is_open ())
    {
      NS_LOG_ERROR ("Cannot open CSV file: " << filename);
      return;
    }
  
  m_trajectory.waypoints.clear ();
  
  std::string line;
  std::getline (file, line); // Skip header
  
  while (std::getline (file, line))
    {
      std::istringstream ss (line);
      std::string field;
      std::vector<std::string> fields;
      
      while (std::getline (ss, field, ','))
        {
          fields.push_back (field);
        }
      
      if (fields.size () >= 13)
        {
          TrajectoryPoint point;
          m_trajectory.vehicleId = fields[0];
          m_trajectory.vehicleType = fields[2];
          
          point.position.x = std::stod (fields[3]);
          point.position.y = std::stod (fields[4]);
          point.position.z = 0.2; // Default height
          
          point.velocity.x = std::stod (fields[5]);
          point.velocity.y = std::stod (fields[6]);
          point.velocity.z = 0.0;
          
          point.time = std::stod (fields[1]);
          point.heading = std::stod (fields[11]);
          
          m_trajectory.waypoints.push_back (point);
        }
    }
  
  file.close ();
  NS_LOG_INFO ("Loaded " << m_trajectory.waypoints.size () << " waypoints from " << filename);
}

} // namespace ns3