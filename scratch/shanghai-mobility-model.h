/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Mobility model for Shanghai dataset trajectories
 */

#ifndef SHANGHAI_MOBILITY_MODEL_H
#define SHANGHAI_MOBILITY_MODEL_H

#include "ns3/mobility-model.h"
#include "ns3/vector.h"
#include "ns3/ptr.h"
#include <vector>
#include <map>
#include <string>

namespace ns3 {

struct TrajectoryPoint
{
  Vector position;
  Vector velocity;
  double time;
  double heading;
};

struct VehicleTrajectory
{
  std::string vehicleId;
  std::string vehicleType;
  std::vector<TrajectoryPoint> waypoints;
  double maxSpeed;
  double length;
  double width;
};

class ShanghaiMobilityModel : public MobilityModel
{
public:
  static TypeId GetTypeId (void);
  ShanghaiMobilityModel ();
  virtual ~ShanghaiMobilityModel ();

  void LoadTrajectoryFromOpenScenario (const std::string& filename);
  void LoadTrajectoryFromCSV (const std::string& filename);
  void SetVehicleTrajectory (const VehicleTrajectory& trajectory);
  void SetSimulationTimeScale (double scale);

private:
  virtual Vector DoGetPosition (void) const;
  virtual void DoSetPosition (const Vector &position);
  virtual Vector DoGetVelocity (void) const;
  virtual void DoInitialize (void);

  void UpdatePosition (void);
  Vector InterpolatePosition (double time) const;
  Vector InterpolateVelocity (double time) const;
  void ParseOpenScenarioFile (const std::string& filename);
  void ParseCSVFile (const std::string& filename);
  TrajectoryPoint ParseWaypoint (const std::string& positionStr) const;

  VehicleTrajectory m_trajectory;
  double m_timeScale;
  double m_startTime;
  EventId m_updateEvent;
  bool m_trajectoryLoaded;
};

} // namespace ns3

#endif /* SHANGHAI_MOBILITY_MODEL_H */