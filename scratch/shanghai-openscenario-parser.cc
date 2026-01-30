/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Enhanced OpenSCENARIO parser implementation for Shanghai dataset
 */

#include "shanghai-openscenario-parser.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include <sstream>
#include <cmath>
#include <fstream>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("ShanghaiOpenScenarioParser");
NS_OBJECT_ENSURE_REGISTERED (ShanghaiOpenScenarioParser);

TypeId
ShanghaiOpenScenarioParser::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::ShanghaiOpenScenarioParser")
    .SetParent<Object> ()
    .SetGroupName ("Applications")
    .AddConstructor<ShanghaiOpenScenarioParser> ();
  return tid;
}

ShanghaiOpenScenarioParser::ShanghaiOpenScenarioParser ()
{
  NS_LOG_FUNCTION (this);
}

ShanghaiOpenScenarioParser::~ShanghaiOpenScenarioParser ()
{
  NS_LOG_FUNCTION (this);
}

ScenarioMetadata
ShanghaiOpenScenarioParser::ParseScenario (const std::string& xoscFile)
{
  NS_LOG_FUNCTION (this << xoscFile);
  
  ScenarioMetadata scenario;
  scenario.scenarioId = xoscFile;
  m_currentScenarioFile = xoscFile;
  
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile (xoscFile.c_str ()) != tinyxml2::XML_SUCCESS)
    {
      NS_LOG_ERROR ("Failed to load OpenSCENARIO file: " << xoscFile);
      return scenario;
    }
  
  tinyxml2::XMLElement* root = doc.FirstChildElement ("OpenSCENARIO");
  if (!root)
    {
      NS_LOG_ERROR ("No OpenSCENARIO root element found");
      return scenario;
    }
  
  // Parse scenario metadata
  tinyxml2::XMLElement* fileHeader = root->FirstChildElement ("FileHeader");
  if (fileHeader)
    {
      const char* description = fileHeader->Attribute ("description");
      if (description)
        {
          scenario.description = std::string (description);
        }
    }
  
  // Parse vehicles
  tinyxml2::XMLElement* entities = root->FirstChildElement ("Entities");
  if (entities)
    {
      tinyxml2::XMLElement* scenarioObject = entities->FirstChildElement ("ScenarioObject");
      while (scenarioObject)
        {
          const char* name = scenarioObject->Attribute ("name");
          if (name)
            {
              std::string vehicleId = std::string (name);
              VehicleInfo vehicle = ParseSingleVehicle (scenarioObject, root);
              scenario.vehicles[vehicleId] = vehicle;
            }
          scenarioObject = scenarioObject->NextSiblingElement ("ScenarioObject");
        }
    }
  
  // Calculate scenario duration from trajectories
  double maxTime = 0.0;
  for (auto& vehiclePair : scenario.vehicles)
    {
      VehicleInfo& vehicle = vehiclePair.second;
      if (!vehicle.waypoints.empty ())
        {
          maxTime = std::max (maxTime, vehicle.waypoints.back ().time);
        }
    }
  scenario.duration = maxTime;
  scenario.totalVehicles = scenario.vehicles.size ();
  
  NS_LOG_INFO ("Parsed scenario: " << scenario.scenarioId 
               << " with " << scenario.totalVehicles 
               << " vehicles, duration: " << scenario.duration << "s");
  
  m_currentScenario = scenario;
  return scenario;
}

VehicleInfo
ShanghaiOpenScenarioParser::ParseSingleVehicle (tinyxml2::XMLElement* scenarioObject, tinyxml2::XMLElement* root)
{
  VehicleInfo vehicle;
  
  // Get vehicle name/ID
  const char* name = scenarioObject->Attribute ("name");
  if (name)
    {
      vehicle.vehicleId = std::string (name);
    }
  
  // Parse vehicle properties
  tinyxml2::XMLElement* entityObject = scenarioObject->FirstChildElement ("EntityObject");
  if (entityObject)
    {
      tinyxml2::XMLElement* catalogReference = entityObject->FirstChildElement ("CatalogReference");
      if (catalogReference)
        {
          const char* entryName = catalogReference->Attribute ("entryName");
          if (entryName)
            {
              vehicle.model = std::string (entryName);
            }
        }
      
      tinyxml2::XMLElement* vehicleElem = entityObject->FirstChildElement ("Vehicle");
      if (vehicleElem)
        {
          const char* vehicleCategory = vehicleElem->Attribute ("vehicleCategory");
          if (vehicleCategory)
            {
              vehicle.vehicleType = std::string (vehicleCategory);
            }
          
          // Parse bounding box
          tinyxml2::XMLElement* boundingBox = vehicleElem->FirstChildElement ("BoundingBox");
          if (boundingBox)
            {
              tinyxml2::XMLElement* dimensions = boundingBox->FirstChildElement ("Dimensions");
              if (dimensions)
                {
                  dimensions->QueryDoubleAttribute ("length", &vehicle.length);
                  dimensions->QueryDoubleAttribute ("width", &vehicle.width);
                  dimensions->QueryDoubleAttribute ("height", &vehicle.height);
                }
            }
        }
    }
  
  // Parse trajectory from Story elements
  tinyxml2::XMLElement* story = root->FirstChildElement ("Story");
  while (story)
    {
      std::vector<tinyxml2::XMLElement*> trajectoryActions = FindAllTrajectoryActions (story, vehicle.vehicleId);
      
      for (tinyxml2::XMLElement* action : trajectoryActions)
        {
          tinyxml2::XMLElement* followTrajectoryAction = action->FirstChildElement ("FollowTrajectoryAction");
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
                          double baseTime = 0.0;
                          
                          while (vertex)
                            {
                              ScenarioWaypoint waypoint = ParseVertex (vertex, baseTime);
                              vehicle.waypoints.push_back (waypoint);
                              
                              // Increment time based on typical frame rate (25 FPS = 0.04s)
                              baseTime += 0.04;
                              
                              vertex = vertex->NextSiblingElement ("Vertex");
                            }
                        }
                    }
                }
            }
        }
      
      story = story->NextSiblingElement ("Story");
    }
  
  // Calculate vehicle statistics
  CalculateVehicleStatistics (vehicle);
  
  return vehicle;
}

ScenarioWaypoint
ShanghaiOpenScenarioParser::ParseVertex (tinyxml2::XMLElement* vertex, double baseTime)
{
  ScenarioWaypoint waypoint;
  waypoint.time = baseTime;
  
  tinyxml2::XMLElement* position = vertex->FirstChildElement ("Position");
  if (position)
    {
      tinyxml2::XMLElement* worldPos = position->FirstChildElement ("WorldPosition");
      if (worldPos)
        {
          worldPos->QueryDoubleAttribute ("x", &waypoint.position.x);
          worldPos->QueryDoubleAttribute ("y", &waypoint.position.y);
          worldPos->QueryDoubleAttribute ("z", &waypoint.position.z);
          worldPos->QueryDoubleAttribute ("h", &waypoint.heading);
          worldPos->QueryDoubleAttribute ("p", &waypoint.speed);
        }
    }
  
  // If speed not directly available, calculate from heading
  tinyxml2::XMLElement* orientation = vertex->FirstChildElement ("Orientation");
  if (orientation && waypoint.speed == 0.0)
    {
      orientation->QueryDoubleAttribute ("h", &waypoint.heading);
    }
  
  return waypoint;
}

std::vector<tinyxml2::XMLElement*>
ShanghaiOpenScenarioParser::FindAllTrajectoryActions (tinyxml2::XMLElement* story, const std::string& vehicleId)
{
  std::vector<tinyxml2::XMLElement*> actions;
  
  tinyxml2::XMLElement* act = story->FirstChildElement ("Act");
  while (act)
    {
      tinyxml2::XMLElement* maneuverGroup = act->FirstChildElement ("ManeuverGroup");
      while (maneuverGroup)
        {
          // Check if this maneuver group targets our vehicle
          tinyxml2::XMLElement* actors = maneuverGroup->FirstChildElement ("Actors");
          if (actors)
            {
              tinyxml2::XMLElement* entityRef = actors->FirstChildElement ("EntityRef");
              while (entityRef)
                {
                  const char* refName = entityRef->Attribute ("entityName");
                  if (refName && std::string (refName) == vehicleId)
                    {
                      // This maneuver group targets our vehicle
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
                                          actions.push_back (action);
                                        }
                                    }
                                  action = action->NextSiblingElement ("Action");
                                }
                              event = event->NextSiblingElement ("Event");
                            }
                          maneuver = maneuver->NextSiblingElement ("Maneuver");
                        }
                    }
                  entityRef = entityRef->NextSiblingElement ("EntityRef");
                }
            }
          maneuverGroup = maneuverGroup->NextSiblingElement ("ManeuverGroup");
        }
      act = act->NextSiblingElement ("Act");
    }
  
  return actions;
}

void
ShanghaiOpenScenarioParser::CalculateVehicleStatistics (VehicleInfo& vehicle)
{
  if (vehicle.waypoints.empty ())
    {
      vehicle.maxSpeed = 0.0;
      vehicle.averageSpeed = 0.0;
      vehicle.totalDistance = 0.0;
      return;
    }
  
  vehicle.totalDistance = 0.0;
  double totalSpeed = 0.0;
  vehicle.maxSpeed = 0.0;
  
  for (size_t i = 1; i < vehicle.waypoints.size (); ++i)
    {
      const ScenarioWaypoint& wp1 = vehicle.waypoints[i-1];
      const ScenarioWaypoint& wp2 = vehicle.waypoints[i];
      
      double distance = CalculateDistance (wp1.position, wp2.position);
      double timeDiff = wp2.time - wp1.time;
      
      if (timeDiff > 0)
        {
          double speed = distance / timeDiff; // m/s
          totalSpeed += speed;
          vehicle.maxSpeed = std::max (vehicle.maxSpeed, speed);
        }
      
      vehicle.totalDistance += distance;
    }
  
  vehicle.averageSpeed = (vehicle.waypoints.size () > 1) ? 
    totalSpeed / (vehicle.waypoints.size () - 1) : 0.0;
}

double
ShanghaiOpenScenarioParser::CalculateDistance (const Vector& p1, const Vector& p2) const
{
  double dx = p2.x - p1.x;
  double dy = p2.y - p1.y;
  double dz = p2.z - p1.z;
  return std::sqrt (dx*dx + dy*dy + dz*dz);
}

std::vector<std::string>
ShanghaiOpenScenarioParser::SelectVehiclesByCriteria (const ScenarioMetadata& scenario, 
                                                     int maxVehicles, 
                                                     double minDuration,
                                                     double minSpeed)
{
  std::vector<std::pair<std::string, double>> candidates;
  
  for (const auto& vehiclePair : scenario.vehicles)
    {
      const VehicleInfo& vehicle = vehiclePair.second;
      
      // Check if vehicle meets criteria
      if (vehicle.waypoints.empty ())
        continue;
        
      double duration = vehicle.waypoints.back ().time - vehicle.waypoints[0].time;
      
      if (duration >= minDuration && vehicle.averageSpeed >= minSpeed)
        {
          // Score based on duration, distance, and speed variety
          double score = duration + vehicle.totalDistance / 100.0 + vehicle.maxSpeed;
          candidates.push_back ({vehiclePair.first, score});
        }
    }
  
  // Sort by score (descending) and select top vehicles
  std::sort (candidates.begin (), candidates.end (), 
             [](const auto& a, const auto& b) { return a.second > b.second; });
  
  std::vector<std::string> selected;
  int count = std::min (maxVehicles, (int)candidates.size ());
  
  for (int i = 0; i < count; ++i)
    {
      selected.push_back (candidates[i].first);
    }
  
  return selected;
}

void
ShanghaiOpenScenarioParser::ExportToNs2Format (const ScenarioMetadata& scenario, 
                                              const std::vector<std::string>& selectedVehicles,
                                              const std::string& outputPath)
{
  std::ofstream file (outputPath);
  if (!file.is_open ())
    {
      NS_LOG_ERROR ("Cannot create NS-2 mobility file: " << outputPath);
      return;
    }
  
  // NS-2 format header
  file << "# NS-2 Mobility trace\n";
  file << "# Generated from Shanghai dataset: " << scenario.scenarioId << "\n";
  file << "# Format: $node_(<id>) set X_ <x>\n";
  file << "#         $node_(<id>) set Y_ <y>\n";
  file << "#         $node_(<id>) set Z_ <z>\n";
  file << "#         $ns_ at <time> \"$node_(<id>) setdest <x> <y> <speed>\"\n\n";
  
  // Map vehicle IDs to sequential node numbers
  std::map<std::string, int> nodeMap;
  int nodeId = 0;
  
  for (const std::string& vehicleId : selectedVehicles)
    {
      nodeMap[vehicleId] = nodeId++;
    }
  
  // Export initial positions and movement commands
  for (const std::string& vehicleId : selectedVehicles)
    {
      auto it = scenario.vehicles.find (vehicleId);
      if (it == scenario.vehicles.end ())
        continue;
        
      const VehicleInfo& vehicle = it->second;
      int nodeId = nodeMap[vehicleId];
      
      if (vehicle.waypoints.empty ())
        continue;
      
      // Initial position
      const ScenarioWaypoint& wp0 = vehicle.waypoints[0];
      file << "$node_(" << nodeId << ") set X_ " << wp0.position.x << "\n";
      file << "$node_(" << nodeId << ") set Y_ " << wp0.position.y << "\n";
      file << "$node_(" << nodeId << ") set Z_ " << wp0.position.z << "\n";
      
      // Movement commands
      for (size_t i = 1; i < vehicle.waypoints.size (); ++i)
        {
          const ScenarioWaypoint& wp = vehicle.waypoints[i];
          double distance = CalculateDistance (vehicle.waypoints[i-1].position, wp.position);
          double timeDiff = wp.time - vehicle.waypoints[i-1].time;
          double speed = (timeDiff > 0) ? (distance / timeDiff) : 0.0;
          
          file << "$ns_ at " << wp.time << " \"$node_(" << nodeId 
               << ") setdest " << wp.position.x << " " << wp.position.y 
               << " " << speed << "\"\n";
        }
      
      file << "\n";
    }
  
  file.close ();
  NS_LOG_INFO ("Exported " << selectedVehicles.size () << " vehicles to NS-2 format: " << outputPath);
}

} // namespace ns3